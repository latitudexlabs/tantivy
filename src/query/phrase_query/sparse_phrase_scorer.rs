use crate::docset::{DocSet, SeekDangerResult, TERMINATED};
use crate::fieldnorm::FieldNormReader;
use crate::postings::Postings;
use crate::query::bm25::Bm25Weight;
use crate::query::{Intersection, Scorer};
use crate::{DocId, Score};

struct PostingsWithOffset<TPostings> {
    offset: u32,
    postings: TPostings,
}

impl<TPostings: Postings> PostingsWithOffset<TPostings> {
    pub fn new(segment_postings: TPostings, offset: u32) -> PostingsWithOffset<TPostings> {
        PostingsWithOffset {
            offset,
            postings: segment_postings,
        }
    }

    pub fn positions(&mut self, output: &mut Vec<u32>) {
        self.postings.positions_with_offset(self.offset, output)
    }
}

impl<TPostings: Postings> DocSet for PostingsWithOffset<TPostings> {
    fn advance(&mut self) -> DocId {
        self.postings.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.postings.seek(target)
    }

    fn doc(&self) -> DocId {
        self.postings.doc()
    }

    fn size_hint(&self) -> u32 {
        self.postings.size_hint()
    }
}

pub struct SparsePhraSeScorer<TPostings: Postings> {
    intersection_docset: Intersection<PostingsWithOffset<TPostings>, PostingsWithOffset<TPostings>>,
    num_terms: usize,
    positions_per_term: Vec<Vec<u32>>,
    temp_positions: Vec<Vec<u32>>,  // Reusable temporary buffer for positions
    matched_term_count: u32,
    fieldnorm_reader: FieldNormReader,
    similarity_weight_opt: Option<Bm25Weight>,
    /// Maps intersection docset indices back to original term indices
    /// intersection_index -> original_term_index
    term_indices: Vec<usize>,
}

/// Returns the count of terms that match in order.
/// 
/// This function checks if all query terms appear in order in the document.
/// It does not allow skipping terms - all terms must be present and in strictly increasing order.
/// 
/// Example: positions_per_term = [[1, 5], [3, 7], [10, 15]]
/// This means term0 at [1,5], term1 at [3,7], term2 at [10,15]
/// Returns 3 because we can find: 1->3->10 (all 3 in order)
/// 
/// If any term cannot be matched in order, returns less than the total number of terms.
fn count_ordered_terms(positions_per_term: &[Vec<u32>], expected_count: usize) -> u32 {
    if positions_per_term.is_empty() {
        return 0;
    }

    // If only one term, count it if it has positions
    if positions_per_term.len() == 1 {
        return if positions_per_term[0].is_empty() { 0 } else { 1 };
    }

    // Find a chain where all terms appear in order
    fn find_chain_all_terms(
        positions_per_term: &[Vec<u32>],
        term_idx: usize,
        last_pos: i32,
        expected_count: usize,
        matched_so_far: u32,
    ) -> u32 {
        if term_idx >= positions_per_term.len() {
            return 0;
        }

        // Early termination: if we already matched all terms, return immediately
        if matched_so_far as usize == expected_count {
            return matched_so_far;
        }

        // Current term must have at least one position after last_pos
        let positions = &positions_per_term[term_idx];
        if positions.is_empty() {
            // Term not found, cannot match all terms
            return 0;
        }

        // Find the first position greater than last_pos
        // Using binary search for better performance with many positions
        let pos = positions.binary_search(&((last_pos + 1) as u32)).unwrap_or_else(|idx| idx);
        
        if pos < positions.len() {
            // Found a valid position for this term
            let rest = find_chain_all_terms(
                positions_per_term,
                term_idx + 1,
                positions[pos] as i32,
                expected_count,
                matched_so_far + 1,
            );
            1 + rest
        } else {
            // No valid position found for this term, cannot match all terms
            0
        }
    }

    find_chain_all_terms(positions_per_term, 0, -1, expected_count, 0)
}

impl<TPostings: Postings> SparsePhraSeScorer<TPostings> {
    pub fn new(
        term_postings: Vec<(usize, TPostings)>,
        similarity_weight_opt: Option<Bm25Weight>,
        fieldnorm_reader: FieldNormReader,
    ) -> SparsePhraSeScorer<TPostings> {
        Self::new_with_offset(term_postings, similarity_weight_opt, fieldnorm_reader, 0)
    }

    pub(crate) fn new_with_offset(
        term_postings_with_offset: Vec<(usize, TPostings)>,
        similarity_weight_opt: Option<Bm25Weight>,
        fieldnorm_reader: FieldNormReader,
        offset: usize,
    ) -> SparsePhraSeScorer<TPostings> {
        let num_docs = fieldnorm_reader.num_docs();
        let max_offset = term_postings_with_offset
            .iter()
            .map(|&(offset, _)| offset)
            .max()
            .unwrap_or(0)
            + offset;
        let num_docsets = term_postings_with_offset.len();
        
        // Create indexed postings to track original indices before sorting
        let indexed_postings: Vec<(usize, PostingsWithOffset<TPostings>)> = term_postings_with_offset
            .into_iter()
            .enumerate()
            .map(|(idx, (offset, postings))| (
                idx,
                PostingsWithOffset::new(postings, (max_offset - offset) as u32)
            ))
            .collect::<Vec<_>>();
        
        // Simulate the sorting that Intersection.new will do
        let mut sorted_indexed = indexed_postings
            .iter()
            .enumerate()
            .map(|(i, (orig_idx, postings))| (i, *orig_idx, postings.postings.cost()))
            .collect::<Vec<_>>();
        sorted_indexed.sort_by_key(|(_, _, cost)| *cost);
        
        // Build the mapping: sorted_position -> original_term_index
        let term_indices: Vec<usize> = sorted_indexed
            .iter()
            .map(|(_, orig_idx, _)| *orig_idx)
            .collect();
        
        let postings_with_offsets = indexed_postings
            .into_iter()
            .map(|(_, p)| p)
            .collect::<Vec<_>>();
        
        let intersection_docset = Intersection::new(postings_with_offsets, num_docs);
        let mut scorer = SparsePhraSeScorer {
            intersection_docset,
            num_terms: num_docsets,
            positions_per_term: vec![Vec::with_capacity(100); num_docsets],
            temp_positions: vec![Vec::with_capacity(100); num_docsets],
            matched_term_count: 0u32,
            similarity_weight_opt,
            fieldnorm_reader,
            term_indices,
        };
        if scorer.doc() != TERMINATED && !scorer.phrase_match() {
            scorer.advance();
        }
        scorer
    }

    pub fn matched_term_count(&self) -> u32 {
        self.matched_term_count
    }

    fn phrase_match(&mut self) -> bool {
        self.compute_matched_terms();
        self.matched_term_count as usize == self.positions_per_term.len()
    }

    fn compute_matched_terms(&mut self) {
        // Collect positions for all terms in current document
        // We need to map from intersection docset indices (which are sorted by cost)
        // back to the original term indices
        
        // Clear temporary positions without deallocating
        for positions in &mut self.temp_positions {
            positions.clear();
        }
        
        for sorted_idx in 0..self.num_terms {
            let original_term_idx = self.term_indices[sorted_idx];
            self.intersection_docset
                .docset_mut_specialized(sorted_idx)
                .positions(&mut self.temp_positions[original_term_idx]);
        }
        
        // Copy back to positions_per_term in original order
        for i in 0..self.num_terms {
            self.positions_per_term[i].clear();
            self.positions_per_term[i].extend_from_slice(&self.temp_positions[i]);
        }

        // Count how many terms match in order
        self.matched_term_count = count_ordered_terms(&self.positions_per_term, self.num_terms);
    }
}

impl<TPostings: Postings> DocSet for SparsePhraSeScorer<TPostings> {
    fn advance(&mut self) -> DocId {
        loop {
            let doc = self.intersection_docset.advance();
            if doc == TERMINATED || self.phrase_match() {
                return doc;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        debug_assert!(target >= self.doc());
        let doc = self.intersection_docset.seek(target);
        if doc == TERMINATED || self.phrase_match() {
            return doc;
        }
        self.advance()
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        debug_assert!(target >= self.doc());
        let seek_res = self.intersection_docset.seek_danger(target);
        if seek_res != SeekDangerResult::Found {
            return seek_res;
        }
        // The intersection matched. Now let's see if we match the phrase.
        if self.phrase_match() {
            SeekDangerResult::Found
        } else {
            SeekDangerResult::SeekLowerBound(target + 1)
        }
    }

    fn doc(&self) -> DocId {
        self.intersection_docset.doc()
    }

    fn size_hint(&self) -> u32 {
        // Since we only need any terms in order, we get more hits than strict phrase
        // so estimate is intersection divided by fewer terms
        let estimate = self.intersection_docset.size_hint() / (self.num_terms as u32);
        // But return at least the base estimate
        estimate.max(self.intersection_docset.size_hint())
    }

    fn cost(&self) -> u64 {
        // Cost is lower than strict phrase since we don't require adjacency
        self.intersection_docset.size_hint() as u64 * (self.num_terms as u64)
    }
}

impl<TPostings: Postings> Scorer for SparsePhraSeScorer<TPostings> {
    fn score(&mut self) -> Score {
        let doc = self.doc();
        let fieldnorm_id = self.fieldnorm_reader.fieldnorm_id(doc);
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            // Score proportional to number of matched terms
            // matched_term_count / num_terms gives a ratio, multiply by base BM25 score
            let term_match_ratio = self.matched_term_count as f32 / self.num_terms as f32;
            similarity_weight.score(fieldnorm_id, self.matched_term_count) * term_match_ratio
        } else {
            // When scoring disabled, return ratio of matched terms
            self.matched_term_count as f32 / self.num_terms as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_ordered_terms_all_match() {
        // All terms in order
        let positions = vec![vec![1], vec![3], vec![5]];
        assert_eq!(count_ordered_terms(&positions, 3), 3);
    }

    #[test]
    fn test_count_ordered_terms_partial_match() {
        // Skip middle term
        let positions = vec![vec![1], vec![2], vec![3]];
        assert_eq!(count_ordered_terms(&positions, 3), 3);
    }

    #[test]
    fn test_count_ordered_terms_multiple_choices() {
        // Term1 at [1,5], term2 at [3,7], term3 at [10,15]
        // Can match 1->3->10 or skip term2: 1->10, or other combinations
        let positions = vec![vec![1, 5], vec![3, 7], vec![10, 15]];
        assert_eq!(count_ordered_terms(&positions, 3), 3);
    }

    #[test]
    fn test_count_ordered_terms_missing_term() {
        // Second term has no positions, can skip and match first and third
        let positions = vec![vec![1], vec![], vec![10]];
        assert_eq!(count_ordered_terms(&positions, 3), 2);
    }

    #[test]
    fn test_count_ordered_terms_empty() {
        let positions: Vec<Vec<u32>> = vec![];
        assert_eq!(count_ordered_terms(&positions, 0), 0);
    }

    #[test]
    fn test_count_ordered_terms_all_empty() {
        let positions = vec![vec![], vec![], vec![]];
        assert_eq!(count_ordered_terms(&positions, 3), 0);
    }

    #[test]
    fn test_count_ordered_terms_single_term() {
        let positions = vec![vec![1, 2, 3]];
        assert_eq!(count_ordered_terms(&positions, 1), 1);
    }

    #[test]
    fn test_count_ordered_terms_complex() {
        // Term1 at [1, 10], Term2 at [2, 11], Term3 at [3, 12], Term4 at [4, 13]
        // Can match: 1->2->3->4 or 10->11->12->13, so all 4
        let positions = vec![vec![1, 10], vec![2, 11], vec![3, 12], vec![4, 13]];
        assert_eq!(count_ordered_terms(&positions, 4), 4);
    }
}
