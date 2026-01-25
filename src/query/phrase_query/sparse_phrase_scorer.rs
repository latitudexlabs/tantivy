use crate::docset::{DocSet, SeekDangerResult, TERMINATED};
use crate::fieldnorm::FieldNormReader;
use crate::postings::Postings;
use crate::query::bm25::Bm25Weight;
use crate::query::Scorer;
use crate::{DocId, Score};

pub struct SparsePhraSeScorer<TPostings: Postings> {
    /// Optional postings for each term (None if term is missing from index)
    term_postings: Vec<Option<TPostings>>,
    /// Tracks which postings exist
    existing_postings_indices: Vec<usize>,
    /// Index of primary postings in term_postings (cached for performance)
    primary_idx: usize,
    /// Current document across all postings
    current_doc: DocId,
    num_terms: usize,
    total_terms: usize,
    positions_per_term: Vec<Vec<u32>>,
    matched_term_count: u32,
    fieldnorm_reader: FieldNormReader,
    similarity_weight_opt: Option<Bm25Weight>,
}

/// Returns the count of terms that match in order, allowing for missing terms (from the index).
/// 
/// A document matches if it contains all PRESENT terms from the query appearing in the correct order.
/// "Missing terms" (terms not found in a specific document due to their positions being empty) 
/// can be skipped. However, if a term IS present in the document but appears out of order
/// relative to other present terms, the match fails and returns 0.
/// 
/// Uses an iterative approach for better performance than recursion.
/// 
/// Example: positions_per_term = [[1, 5], [], [10, 15]]
/// This means term0 at [1,5], term1 missing, term2 at [10,15]
/// Returns 2 because term0(1) < term2(10), skipping the missing term1
/// 
/// Example: positions_per_term = [[1], [0], [2]]
/// term0(1), term1(0), term2(2) - term1 is out of order relative to term0
/// Returns 0 because not all terms can be matched in order
fn count_ordered_terms(positions_per_term: &[Vec<u32>]) -> u32 {
    if positions_per_term.is_empty() {
        return 0;
    }

    let mut last_pos: i32 = -1;
    let mut matched_count = 0u32;

    for positions in positions_per_term {
        if positions.is_empty() {
            // Term is missing in this document, skip it and continue
            continue;
        }

        // Term is present in the document
        // Find the first position greater than last_pos
        let search_val = (last_pos + 1) as u32;
        let pos = positions.binary_search(&search_val).unwrap_or_else(|idx| idx);

        if pos < positions.len() {
            // Found a valid position for this term that maintains order
            last_pos = positions[pos] as i32;
            matched_count += 1;
        } else {
            // No valid position found for this term
            // This means the document is out of order
            return 0;
        }
    }

    matched_count
}

impl<TPostings: Postings> SparsePhraSeScorer<TPostings> {
    pub fn new(
        term_postings_with_option: Vec<(usize, Option<TPostings>)>,
        total_terms: usize,
        similarity_weight_opt: Option<Bm25Weight>,
        fieldnorm_reader: FieldNormReader,
    ) -> SparsePhraSeScorer<TPostings> {
        // Extract just the postings, tracking which indices have actual postings
        let mut term_postings = Vec::new();
        let mut existing_postings_indices = Vec::new();
        
        for (idx, (_, postings_opt)) in term_postings_with_option.into_iter().enumerate() {
            if postings_opt.is_some() {
                existing_postings_indices.push(idx);
            }
            term_postings.push(postings_opt);
        }
        
        let num_terms = term_postings.len();
        let primary_idx = if existing_postings_indices.is_empty() {
            0
        } else {
            existing_postings_indices[0]
        };
        
        let mut scorer = SparsePhraSeScorer {
            term_postings,
            existing_postings_indices,
            primary_idx,
            current_doc: TERMINATED,
            num_terms,
            total_terms,
            positions_per_term: vec![Vec::with_capacity(100); num_terms],
            matched_term_count: 0u32,
            fieldnorm_reader,
            similarity_weight_opt,
        };
        
        // Initialize to first valid document
        if !scorer.existing_postings_indices.is_empty() {
            scorer.advance();
        }
        
        scorer
    }

    pub fn matched_term_count(&self) -> u32 {
        self.matched_term_count
    }

    fn phrase_match(&mut self, doc: DocId) -> bool {
        let expected_count = self.existing_postings_indices.len();
        // Early termination: if no existing postings, no match
        if expected_count == 0 {
            return false;
        }
        
        self.compute_matched_terms(doc);
        // Match only if all existing terms (that have postings) are matched in order
        self.matched_term_count as usize == expected_count
    }

    fn compute_matched_terms(&mut self, doc: DocId) {
        // Collect positions for all terms in current document
        for positions in &mut self.positions_per_term {
            positions.clear();
        }
        
        for (idx, postings_opt) in self.term_postings.iter_mut().enumerate() {
            if let Some(postings) = postings_opt {
                if postings.doc() == doc {
                    postings.positions(&mut self.positions_per_term[idx]);
                }
            }
        }

        // Count how many terms match in order (allowing missing terms)
        self.matched_term_count = count_ordered_terms(&self.positions_per_term);
    }
}

impl<TPostings: Postings> DocSet for SparsePhraSeScorer<TPostings> {
    fn advance(&mut self) -> DocId {
        if self.existing_postings_indices.is_empty() {
            self.current_doc = TERMINATED;
            return TERMINATED;
        }

        // Find the next document where all existing terms appear and phrase matches
        // Use cached primary_idx for performance
        let primary_idx = self.primary_idx;
        
        // For the first call, we need to sync all terms to the first common document
        // For subsequent calls, we advance and then sync
        let mut doc = if self.current_doc == TERMINATED {
            // First call: seek all postings to doc 0 to check if it matches
            for &idx in &self.existing_postings_indices {
                if let Some(postings) = &mut self.term_postings[idx] {
                    let _ = postings.seek(0);
                }
            }
            0
        } else {
            // Subsequent calls: advance primary
            if let Some(postings) = &mut self.term_postings[primary_idx] {
                postings.advance()
            } else {
                TERMINATED
            }
        };
        
        loop {
            if doc == TERMINATED {
                self.current_doc = TERMINATED;
                return TERMINATED;
            }
            
            // Sync all other existing terms to this doc
            loop {
                let mut max_doc = doc;
                let mut all_synced = true;
                
                // Check if all other existing terms are at or past this doc
                for &idx in &self.existing_postings_indices[1..] {
                    if let Some(postings) = &mut self.term_postings[idx] {
                        let other_doc = postings.seek(doc);
                        if other_doc == TERMINATED {
                            self.current_doc = TERMINATED;
                            return TERMINATED;
                        }
                        if other_doc > max_doc {
                            max_doc = other_doc;
                            all_synced = false;
                        }
                    }
                }
                
                if all_synced {
                    // All terms are at this document
                    break;
                }
                
                // Some terms are ahead, need to seek primary to catch up
                let new_doc = if let Some(postings) = &mut self.term_postings[primary_idx] {
                    postings.seek(max_doc)
                } else {
                    TERMINATED
                };
                
                if new_doc == TERMINATED || new_doc > max_doc {
                    self.current_doc = TERMINATED;
                    return TERMINATED;
                }
                
                doc = new_doc;
            }
            
            // All existing terms are now at the same document
            // Check if the phrase matches
            if self.phrase_match(doc) {
                self.current_doc = doc;
                return doc;
            }
            
            // Try next document
            if let Some(postings) = &mut self.term_postings[primary_idx] {
                doc = postings.advance();
            } else {
                self.current_doc = TERMINATED;
                return TERMINATED;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        debug_assert!(target >= self.current_doc);
        
        if self.existing_postings_indices.is_empty() {
            self.current_doc = TERMINATED;
            return TERMINATED;
        }

        // Seek primary to target
        // Use cached primary_idx for performance
        let primary_idx = self.primary_idx;
        let mut doc = if let Some(postings) = &mut self.term_postings[primary_idx] {
            postings.seek(target)
        } else {
            TERMINATED
        };
        
        if doc == TERMINATED {
            self.current_doc = TERMINATED;
            return TERMINATED;
        }
        
        loop {
            // Seek other terms to current document  
            let mut max_doc = doc;
            let mut all_at_doc = true;
            
            for &idx in &self.existing_postings_indices[1..] {
                if let Some(postings) = &mut self.term_postings[idx] {
                    let other_doc = postings.seek(doc);
                    if other_doc == TERMINATED {
                        self.current_doc = TERMINATED;
                        return TERMINATED;
                    }
                    if other_doc > max_doc {
                        max_doc = other_doc;
                        all_at_doc = false;
                    }
                }
            }
            
            if all_at_doc {
                // All terms at this document
                if self.phrase_match(doc) {
                    self.current_doc = doc;
                    return doc;
                }
                // Try next document
                if let Some(postings) = &mut self.term_postings[primary_idx] {
                    doc = postings.advance();
                } else {
                    self.current_doc = TERMINATED;
                    return TERMINATED;
                }
                if doc == TERMINATED {
                    self.current_doc = TERMINATED;
                    return TERMINATED;
                }
            } else {
                // Some terms are ahead, seek primary to max_doc
                if let Some(postings) = &mut self.term_postings[primary_idx] {
                    doc = postings.seek(max_doc);
                } else {
                    self.current_doc = TERMINATED;
                    return TERMINATED;
                }
                if doc == TERMINATED {
                    self.current_doc = TERMINATED;
                    return TERMINATED;
                }
            }
        }
    }

    fn seek_danger(&mut self, target: DocId) -> SeekDangerResult {
        match self.seek(target) {
            TERMINATED => SeekDangerResult::SeekLowerBound(target),
            found_doc => {
                if found_doc >= target {
                    SeekDangerResult::Found
                } else {
                    SeekDangerResult::SeekLowerBound(found_doc + 1)
                }
            }
        }
    }

    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        if !self.existing_postings_indices.is_empty() {
            if let Some(postings) = &self.term_postings[self.primary_idx] {
                return postings.size_hint() / (10 * self.num_terms as u32).max(1);
            }
        }
        0
    }

    fn cost(&self) -> u64 {
        if !self.existing_postings_indices.is_empty() {
            if let Some(postings) = &self.term_postings[self.primary_idx] {
                return postings.size_hint() as u64 * 10 * self.num_terms as u64;
            }
        }
        0
    }
}

impl<TPostings: Postings> Scorer for SparsePhraSeScorer<TPostings> {
    fn score(&mut self) -> Score {
        let doc = self.doc();
        let fieldnorm_id = self.fieldnorm_reader.fieldnorm_id(doc);
        
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            // Score proportional to number of matched terms
            // matched_term_count / total_terms gives a ratio, multiply by base BM25 score
            let term_match_ratio = self.matched_term_count as f32 / self.total_terms as f32;
            similarity_weight.score(fieldnorm_id, self.matched_term_count) * term_match_ratio
        } else {
            // When scoring disabled, return ratio of matched terms
            self.matched_term_count as f32 / self.total_terms as f32
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
