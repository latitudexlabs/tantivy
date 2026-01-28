use crate::docset::{DocSet, SeekDangerResult, TERMINATED};
use crate::fieldnorm::FieldNormReader;
use crate::postings::Postings;
use crate::query::bm25::Bm25Weight;
use crate::query::Scorer;
use crate::{DocId, Score};

struct PostingsWithOffset<TPostings> {
    offset: u32,
    postings: TPostings,
}

impl<TPostings: Postings> PostingsWithOffset<TPostings> {
    pub fn new(
        segment_postings: TPostings,
        offset: u32,
        _original_index: usize,
    ) -> PostingsWithOffset<TPostings> {
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
    postings_with_offset: Vec<PostingsWithOffset<TPostings>>,
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

        let postings_with_offset: Vec<PostingsWithOffset<TPostings>> = term_postings_with_offset
            .into_iter()
            .map(|(term_offset, postings)| {
                // Normalize so that earlier offsets get larger added value, preserving ordering.
                PostingsWithOffset::new(postings, (max_offset - term_offset) as u32, term_offset)
            })
            .collect();

        let postings_with_offset = postings_with_offset;

        let num_terms = postings_with_offset.len();
        let mut scorer = SparsePhraSeScorer {
            postings_with_offset,
            num_terms,
            positions_per_term: vec![Vec::with_capacity(100); num_terms],
            matched_term_count: 0u32,
            similarity_weight_opt,
            fieldnorm_reader,
            current_doc: TERMINATED,
        };

        scorer.current_doc = scorer.find_next_doc(0);
        if scorer.current_doc != TERMINATED {
            scorer.compute_matched_terms();
            if scorer.matched_term_count < 2 {
                scorer.advance();
            }
        }
        scorer
    }

    pub fn matched_term_count(&self) -> u32 {
        self.matched_term_count
    }

    fn find_next_doc(&mut self, target: DocId) -> DocId {
        let mut next_doc = TERMINATED;
        for postings in &mut self.postings_with_offset {
            let doc = if postings.doc() < target {
                postings.seek(target)
            } else {
                postings.doc()
            };
            if doc < next_doc {
                next_doc = doc;
            }
        }
        next_doc
    }

    fn compute_matched_terms(&mut self) {
        if self.current_doc == TERMINATED {
            self.matched_term_count = 0;
            return;
        }

        for (i, postings) in self.postings_with_offset.iter_mut().enumerate() {
            self.positions_per_term[i].clear();
            if postings.doc() == self.current_doc {
                postings.positions(&mut self.positions_per_term[i]);
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
        loop {
            self.current_doc = match self.current_doc {
                TERMINATED => TERMINATED,
                doc => self.find_next_doc(doc + 1),
            };
            if self.current_doc == TERMINATED {
                return TERMINATED;
            }
            self.compute_matched_terms();
            if self.matches_doc() {
                return self.current_doc;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        debug_assert!(target >= self.doc());
        self.current_doc = self.find_next_doc(target);
        if self.current_doc == TERMINATED {
            return TERMINATED;
        }
        self.compute_matched_terms();
        if self.matches_doc() {
            self.current_doc
        } else {
            self.advance()
        }
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
        self.postings_with_offset
            .iter()
            .map(DocSet::size_hint)
            .max()
            .unwrap_or(0)
    }

    /// Returns a best-effort hint of the
    /// cost to drive the docset.
    fn cost(&self) -> u64 {
        self.postings_with_offset
            .iter()
            .map(DocSet::cost)
            .max()
            .unwrap_or(0)
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
