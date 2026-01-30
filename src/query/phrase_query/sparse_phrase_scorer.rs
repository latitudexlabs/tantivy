use crate::docset::{DocSet, SeekDangerResult, TERMINATED};
use crate::fieldnorm::FieldNormReader;
use crate::postings::Postings;
use crate::query::bm25::Bm25Weight;
use crate::query::Scorer;
use crate::{DocId, Score};

struct PostingsWithOffset<TPostings> {
    offset: u32,
    postings: TPostings,
    original_index: usize,  // Track which term this was originally
}

impl<TPostings: Postings> PostingsWithOffset<TPostings> {
    pub fn new(
        segment_postings: TPostings,
        offset: u32,
        original_index: usize,
    ) -> PostingsWithOffset<TPostings> {
        PostingsWithOffset {
            offset,
            postings: segment_postings,
            original_index,
        }
    }

    pub fn positions(&mut self, output: &mut Vec<u32>) {
        self.postings.positions_with_offset(self.offset, output)
    }

    pub fn original_index(&self) -> usize {
        self.original_index
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

impl<TPostings: Postings> Postings for PostingsWithOffset<TPostings> {
    fn term_freq(&self) -> u32 {
        self.postings.term_freq()
    }

    fn positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>) {
        self.postings.positions_with_offset(offset + self.offset, output)
    }

    fn append_positions_with_offset(&mut self, offset: u32, output: &mut Vec<u32>) {
        self.postings.append_positions_with_offset(offset + self.offset, output)
    }
}

pub struct SparsePhraSeScorer<TPostings: Postings> {
    // All postings - we'll use the first (smallest) as primary driver
    all_postings: Vec<PostingsWithOffset<TPostings>>,
    num_terms: usize,
    positions_per_term: Vec<Vec<u32>>,
    matched_term_count: u32,
    fieldnorm_reader: FieldNormReader,
    similarity_weight_opt: Option<Bm25Weight>,
    current_doc: DocId,
}

/// Returns the count of terms that match in order.
///
/// This function finds the longest sequence of query terms that appear in order in the document,
/// allowing some terms to be missing or non-consecutive.
///
/// Example: positions_per_term = [[1, 5], [3, 7], [10, 15]]
/// This means term0 at [1,5], term1 at [3,7], term2 at [10,15]
/// We can match: 1->3->10 (all 3) or skip term1: 1->10 (2 terms), or skip term0: 3->10 (2 terms), etc.
/// We want the maximum, so return 3.
fn count_ordered_terms(positions_per_term: &[Vec<u32>]) -> u32 {
    if positions_per_term.is_empty() {
        return 0;
    }

    // If only one term, count it if it has positions
    if positions_per_term.len() == 1 {
        return if positions_per_term[0].is_empty() {
            0
        } else {
            1
        };
    }

    // Recursively find the longest subsequence where positions are strictly increasing.
    // Missing terms may be skipped, but terms present in the document must respect the order.
    fn find_longest_chain(positions_per_term: &[Vec<u32>], term_idx: usize, last_pos: i32) -> u32 {
        if term_idx >= positions_per_term.len() {
            return 0;
        }

        let positions = &positions_per_term[term_idx];
        let mut best = 0u32;
        let mut has_progress = false;

        for &pos in positions.iter().filter(|&&pos| (pos as i32) > last_pos) {
            has_progress = true;
            let rest = find_longest_chain(positions_per_term, term_idx + 1, pos as i32);
            best = best.max(1 + rest);
        }

        if positions.is_empty() {
            // Term missing in the document: skipping is allowed.
            best = best.max(find_longest_chain(positions_per_term, term_idx + 1, last_pos));
        } else if !has_progress {
            // Term is present but only before `last_pos`; order cannot be satisfied.
            return 0;
        }

        best
    }

    find_longest_chain(positions_per_term, 0, -1)
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
        let max_offset = term_postings_with_offset
            .iter()
            .map(|&(offset, _)| offset)
            .max()
            .unwrap_or(0)
            + offset;

        let mut all_postings: Vec<PostingsWithOffset<TPostings>> = term_postings_with_offset
            .into_iter()
            .enumerate()
            .map(|(idx, (term_offset, postings))| {
                // Normalize so that earlier offsets get larger added value, preserving ordering.
                PostingsWithOffset::new(postings, (max_offset - term_offset) as u32, idx)
            })
            .collect();

        let num_terms = all_postings.len();

        // Sort by size - smallest first for efficient iteration
        all_postings.sort_by_key(|p| p.size_hint());

        let mut scorer = SparsePhraSeScorer {
            all_postings,
            num_terms,
            positions_per_term: vec![Vec::with_capacity(100); num_terms],
            matched_term_count: 0u32,
            similarity_weight_opt,
            fieldnorm_reader,
            current_doc: TERMINATED,
        };

        // Find first matching doc
        scorer.current_doc = scorer.find_next_matching_doc(0);
        scorer
    }

    /// Find the next document that matches the sparse phrase criteria.
    /// Optimized: Uses efficient seeking instead of naive iteration.
    fn find_next_matching_doc(&mut self, mut start_from: DocId) -> DocId {
        'outer: loop {
            // Find minimum doc >= start_from across all postings
            let mut min_doc = TERMINATED;
            for postings in &mut self.all_postings {
                let doc = if postings.doc() < start_from {
                    postings.seek(start_from)
                } else {
                    postings.doc()
                };
                if doc < min_doc {
                    min_doc = doc;
                }
            }

            if min_doc == TERMINATED {
                return TERMINATED;
            }

            // Check if this candidate matches our criteria
            // First, quickly count how many postings are at this doc
            let mut count_at_doc = 0u32;
            for postings in &self.all_postings {
                if postings.doc() == min_doc {
                    count_at_doc += 1;
                    if count_at_doc >= 2 {
                        // Early exit: we have at least 2 terms, worth checking positions
                        break;
                    }
                }
            }

            if count_at_doc < 2 {
                // Not enough terms at this doc, skip it
                start_from = min_doc + 1;
                continue 'outer;
            }

            // Now check positions
            self.compute_matched_terms_for(min_doc);
            if self.matches_doc() {
                return min_doc;
            }

            // Move to next candidate
            start_from = min_doc + 1;
        }
    }

    pub fn matched_term_count(&self) -> u32 {
        self.matched_term_count
    }

    fn compute_matched_terms_for(&mut self, target_doc: DocId) {
        if target_doc == TERMINATED {
            self.matched_term_count = 0;
            return;
        }

        // Clear all position vectors
        for positions in &mut self.positions_per_term {
            positions.clear();
        }

        // Get positions from all postings for this doc
        for postings in &mut self.all_postings {
            let doc = if postings.doc() < target_doc {
                postings.seek(target_doc)
            } else {
                postings.doc()
            };
            
            if doc == target_doc {
                let orig_idx = postings.original_index();
                postings.positions(&mut self.positions_per_term[orig_idx]);
            }
        }

        self.matched_term_count = count_ordered_terms(&self.positions_per_term);
    }


    fn matches_doc(&self) -> bool {
        let all_terms_present = self
            .positions_per_term
            .iter()
            .all(|positions| !positions.is_empty());

        if all_terms_present {
            // If all query terms occur in the doc, they must appear in-order.
            self.matched_term_count as usize == self.num_terms
        } else {
            // Otherwise accept partial matches of length >= 2 in order.
            self.matched_term_count >= 2
        }
    }
}

impl<TPostings: Postings> DocSet for SparsePhraSeScorer<TPostings> {
    fn advance(&mut self) -> DocId {
        self.current_doc = self.find_next_matching_doc(self.current_doc + 1);
        self.current_doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        debug_assert!(target >= self.doc());
        self.current_doc = self.find_next_matching_doc(target);
        self.current_doc
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        debug_assert!(target >= self.doc());
        let doc = self.seek(target);
        if doc == target {
            SeekDangerResult::Found
        } else if doc == TERMINATED {
            SeekDangerResult::SeekLowerBound(TERMINATED)
        } else {
            SeekDangerResult::SeekLowerBound(doc)
        }
    }

    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        // Best estimate is the smallest posting list size
        self.all_postings.first().map(|p| p.size_hint()).unwrap_or(0)
    }

    fn cost(&self) -> u64 {
        // Cost is driven by the smallest posting list
        self.all_postings.first().map(|p| p.cost()).unwrap_or(0)
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
        assert_eq!(count_ordered_terms(&positions), 3);
    }

    #[test]
    fn test_count_ordered_terms_partial_match() {
        // Skip middle term
        let positions = vec![vec![1], vec![2], vec![3]];
        assert_eq!(count_ordered_terms(&positions), 3);
    }

    #[test]
    fn test_count_ordered_terms_multiple_choices() {
        // Term1 at [1,5], term2 at [3,7], term3 at [10,15]
        // Can match 1->3->10 or skip term2: 1->10, or other combinations
        let positions = vec![vec![1, 5], vec![3, 7], vec![10, 15]];
        assert_eq!(count_ordered_terms(&positions), 3);
    }

    #[test]
    fn test_count_ordered_terms_missing_term() {
        // Second term has no positions, can skip and match first and third
        let positions = vec![vec![1], vec![], vec![10]];
        assert_eq!(count_ordered_terms(&positions), 2);
    }

    #[test]
    fn test_count_ordered_terms_empty() {
        let positions: Vec<Vec<u32>> = vec![];
        assert_eq!(count_ordered_terms(&positions), 0);
    }

    #[test]
    fn test_count_ordered_terms_all_empty() {
        let positions = vec![vec![], vec![], vec![]];
        assert_eq!(count_ordered_terms(&positions), 0);
    }

    #[test]
    fn test_count_ordered_terms_single_term() {
        let positions = vec![vec![1, 2, 3]];
        assert_eq!(count_ordered_terms(&positions), 1);
    }

    #[test]
    fn test_count_ordered_terms_complex() {
        // Term1 at [1, 10], Term2 at [2, 11], Term3 at [3, 12], Term4 at [4, 13]
        // Can match: 1->2->3->4 or 10->11->12->13, so all 4
        let positions = vec![vec![1, 10], vec![2, 11], vec![3, 12], vec![4, 13]];
        assert_eq!(count_ordered_terms(&positions), 4);
    }
}
