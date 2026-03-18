use crate::docset::DocSet;
use crate::fieldnorm::FieldNormReader;
use crate::postings::Postings;
use crate::query::bm25::Bm25Weight;
use crate::query::Scorer;
use crate::{DocId, Score};

pub struct FuzzyPhraseScorer<TPostings: Postings> {
    // Boolean scorer to find docs with at least min_match terms
    boolean_scorer: Box<dyn Scorer>,
    // List of postings for each term to check positions
    phrase_postings: Vec<TPostings>,
    min_match: usize,
    match_count: usize,
    fieldnorm_reader: FieldNormReader,
    similarity_weight_opt: Option<Bm25Weight>,
    // Current doc position (TERMINATED if not yet positioned)
    current_doc: DocId,
    // Reusable position buffers to avoid allocations
    position_buffers: Vec<Vec<u32>>,
}

impl<TPostings: Postings> FuzzyPhraseScorer<TPostings> {
    pub fn new(
        boolean_scorer: Box<dyn Scorer>,
        phrase_postings: Vec<TPostings>,
        min_match: usize,
        similarity_weight_opt: Option<Bm25Weight>,
        fieldnorm_reader: FieldNormReader,
    ) -> FuzzyPhraseScorer<TPostings> {
        assert!(phrase_postings.len() >= 2);
        assert!(min_match > 0 && min_match <= phrase_postings.len());
        
        let num_terms = phrase_postings.len();
        let mut scorer = FuzzyPhraseScorer {
            boolean_scorer,
            phrase_postings,
            min_match,
            match_count: 0,
            fieldnorm_reader,
            similarity_weight_opt,
            current_doc: crate::TERMINATED,
            // Pre-allocate position buffers for reuse
            position_buffers: vec![Vec::new(); num_terms],
        };
        
        // The boolean_scorer starts at doc 0 (or TERMINATED if no candidates).
        // We need to check if doc 0 matches, and if not, advance to the first match.
        let initial_doc = scorer.boolean_scorer.doc();
        if initial_doc != crate::TERMINATED {
            scorer.match_count = scorer.count_ordered_matches();
            if scorer.match_count >= scorer.min_match {
                scorer.current_doc = initial_doc;
            } else {
                scorer.current_doc = scorer.advance();
            }
        } else {
            scorer.current_doc = crate::TERMINATED;
        }
        
        scorer
    }

    /// Count how many terms appear in order in the current document.
    /// Only checks postings that are actually at the current document.
    fn count_ordered_matches(&mut self) -> usize {
        let current_doc = self.boolean_scorer.doc();
        let total_terms = self.phrase_postings.len();
        
        // Reuse position buffers - clear them first
        for buffer in &mut self.position_buffers {
            buffer.clear();
        }
        
        let mut terms_found = 0;
        
        for (idx, postings) in self.phrase_postings.iter_mut().enumerate() {
            let postings_doc = postings.doc();
            
            // Check if this posting is at the current document
            let is_present = if postings_doc == current_doc {
                true
            } else if postings_doc < current_doc {
                // Only seek if we're behind
                postings.seek(current_doc) == current_doc
            } else {
                // postings_doc > current_doc - term is not in this doc
                false
            };
            
            if is_present {
                // Reuse pre-allocated buffer
                postings.positions(&mut self.position_buffers[idx]);
                terms_found += 1;
            }
            // Early exit: if we can't possibly reach min_match
            let remaining_terms = total_terms - idx - 1;
            if terms_found + remaining_terms < self.min_match {
                return terms_found;
            }
        }

        // Find the longest chain of terms appearing in order
        self.find_longest_ordered_chain()
    }

    /// Find the longest chain of terms that appear in order, respecting original term sequence.
    /// Optimized with early exit when min_match is reached.
    fn find_longest_ordered_chain(&self) -> usize {
        let all_positions = &self.position_buffers;
        
        // Find first non-empty term
        let mut start_idx = 0;
        while start_idx < all_positions.len() && all_positions[start_idx].is_empty() {
            start_idx += 1;
        }
        
        if start_idx >= all_positions.len() {
            return 0;
        }

        let mut max_depth = 0;
        
        // Try starting from each position in each term with positions
        for term_idx in start_idx..all_positions.len() {
            if all_positions[term_idx].is_empty() {
                continue;
            }
            
            // For each starting position in this term, try to build a chain
            for &start_pos in &all_positions[term_idx] {
                let mut depth = 1;
                let mut last_pos = start_pos;
                
                // Try to extend the chain with subsequent terms in the query order
                for next_idx in (term_idx + 1)..all_positions.len() {
                    let next_positions = &all_positions[next_idx];
                    
                    if next_positions.is_empty() {
                        continue;
                    }
                    
                    // Binary search for first position after last_pos
                    // Since positions are sorted, partition_point is more efficient than binary_search
                    let idx = next_positions.partition_point(|&pos| pos <= last_pos);
                    if idx < next_positions.len() {
                        depth += 1;
                        last_pos = next_positions[idx];
                        
                        // Early exit: we found enough matches
                        if depth >= self.min_match {
                            return depth;
                        }
                    }
                }
                
                max_depth = max_depth.max(depth);
                
                // Early exit if we found enough matches
                if max_depth >= self.min_match {
                    return max_depth;
                }
            }
        }

        max_depth
    }

    #[inline]
    pub(crate) fn match_count(&self) -> u32 {
        self.match_count as u32
    }
}

impl<TPostings: Postings> DocSet for FuzzyPhraseScorer<TPostings> {
    fn advance(&mut self) -> DocId {
        // Use boolean scorer to find next candidate document
        loop {
            let doc = self.boolean_scorer.advance();
            
            if doc == crate::TERMINATED {
                self.current_doc = crate::TERMINATED;
                return crate::TERMINATED;
            }
            
            // Check if terms appear in order
            self.match_count = self.count_ordered_matches();
            
            if self.match_count >= self.min_match {
                self.current_doc = doc;
                return doc;
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let doc = self.boolean_scorer.seek(target);
        
        if doc == crate::TERMINATED {
            self.current_doc = crate::TERMINATED;
            return crate::TERMINATED;
        }
        
        // Check if this document has terms in order
        self.match_count = self.count_ordered_matches();
        
        if self.match_count >= self.min_match {
            self.current_doc = doc;
            return doc;
        }
        
        // Not a match, continue advancing
        self.advance()
    }

    fn seek_danger(&mut self, target: DocId) -> crate::docset::SeekDangerResult {
        use crate::docset::SeekDangerResult;
        
        debug_assert!(
            target >= self.doc(),
            "target ({}) should be greater than or equal to doc ({})",
            target,
            self.doc()
        );
        
        let seek_res = self.boolean_scorer.seek_danger(target);
        if seek_res != SeekDangerResult::Found {
            return seek_res;
        }
        
        // The boolean scorer matched at target. Now check if terms appear in order.
        self.match_count = self.count_ordered_matches();
        
        if self.match_count >= self.min_match {
            self.current_doc = target;
            SeekDangerResult::Found
        } else {
            // Target doc doesn't satisfy order constraint, return lower bound
            SeekDangerResult::SeekLowerBound(target + 1)
        }
    }

    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        // We filter the boolean scorer's results, so we might have fewer matches
        // Return 0 to indicate unknown count
        0
    }
}

impl<TPostings: Postings> Scorer for FuzzyPhraseScorer<TPostings> {
    fn score(&mut self) -> Score {
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            let fieldnorm_id = self.fieldnorm_reader.fieldnorm_id(self.doc());
            similarity_weight.score(fieldnorm_id, self.match_count as u32)
        } else {
            1.0
        }
    }
}
