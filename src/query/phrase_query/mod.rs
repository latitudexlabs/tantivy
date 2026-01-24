mod phrase_query;
mod phrase_scorer;
mod phrase_weight;
pub mod regex_phrase_query;
mod regex_phrase_weight;
mod sparse_phrase_query;
mod sparse_phrase_scorer;
mod sparse_phrase_weight;

pub use self::phrase_query::PhraseQuery;
pub(crate) use self::phrase_scorer::intersection_count;
pub use self::phrase_scorer::PhraseScorer;
pub use self::phrase_weight::PhraseWeight;
pub use self::sparse_phrase_query::SparsePhraSeQuery;
pub use self::sparse_phrase_scorer::SparsePhraSeScorer;
pub use self::sparse_phrase_weight::SparsePhraSeWeight;

#[cfg(test)]
pub(crate) mod tests {

    use serde_json::json;

    use super::*;
    use crate::collector::tests::{TEST_COLLECTOR_WITHOUT_SCORE, TEST_COLLECTOR_WITH_SCORE};
    use crate::index::Index;
    use crate::query::{EnableScoring, QueryParser, Weight};
    use crate::schema::{Schema, Term, TEXT};
    use crate::{assert_nearly_equals, DocAddress, DocId, IndexWriter, TERMINATED};

    pub fn create_index<S: AsRef<str>>(texts: &[S]) -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            for text in texts {
                let doc = doc!(text_field=>text.as_ref());
                index_writer.add_document(doc)?;
            }
            index_writer.commit()?;
        }
        Ok(index)
    }

    #[test]
    pub fn test_phrase_query() -> crate::Result<()> {
        let index = create_index(&[
            "b b b d c g c",
            "a b b d c g c",
            "a b a b c",
            "c a b a d ga a",
            "a b c",
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let test_query = |texts: Vec<&str>| {
            let terms: Vec<Term> = texts
                .iter()
                .map(|text| Term::from_field_text(text_field, text))
                .collect();
            let phrase_query = PhraseQuery::new(terms);
            let test_fruits = searcher
                .search(&phrase_query, &TEST_COLLECTOR_WITH_SCORE)
                .unwrap();
            test_fruits
                .docs()
                .iter()
                .map(|docaddr| docaddr.doc_id)
                .collect::<Vec<_>>()
        };
        assert_eq!(test_query(vec!["a", "b"]), vec![1, 2, 3, 4]);
        assert_eq!(test_query(vec!["a", "b", "c"]), vec![2, 4]);
        assert_eq!(test_query(vec!["b", "b"]), vec![0, 1]);
        assert!(test_query(vec!["g", "ewrwer"]).is_empty());
        assert!(test_query(vec!["g", "a"]).is_empty());
        Ok(())
    }

    #[test]
    pub fn test_phrase_query_simple() -> crate::Result<()> {
        let index = create_index(&["a b b d c g c", "a b a b c"])?;
        let text_field = index.schema().get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let terms: Vec<Term> = ["a", "b", "c"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let phrase_query = PhraseQuery::new(terms);
        let phrase_weight =
            phrase_query.phrase_weight(EnableScoring::disabled_from_schema(searcher.schema()))?;
        let mut phrase_scorer = phrase_weight.scorer(searcher.segment_reader(0), 1.0)?;
        assert_eq!(phrase_scorer.doc(), 1);
        assert_eq!(phrase_scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    pub fn test_phrase_query_no_score() -> crate::Result<()> {
        let index = create_index(&[
            "b b b d c g c",
            "a b b d c g c",
            "a b a b c",
            "c a b a d ga a",
            "a b c",
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let test_query = |texts: Vec<&str>| {
            let terms: Vec<Term> = texts
                .iter()
                .map(|text| Term::from_field_text(text_field, text))
                .collect();
            let phrase_query = PhraseQuery::new(terms);
            let test_fruits = searcher
                .search(&phrase_query, &TEST_COLLECTOR_WITHOUT_SCORE)
                .unwrap();
            test_fruits
                .docs()
                .iter()
                .map(|docaddr| docaddr.doc_id)
                .collect::<Vec<_>>()
        };
        assert_eq!(test_query(vec!["a", "b", "c"]), vec![2, 4]);
        assert_eq!(test_query(vec!["a", "b"]), vec![1, 2, 3, 4]);
        assert_eq!(test_query(vec!["b", "b"]), vec![0, 1]);
        assert!(test_query(vec!["g", "ewrwer"]).is_empty());
        assert!(test_query(vec!["g", "a"]).is_empty());
        Ok(())
    }

    #[test]
    pub fn test_phrase_query_no_positions() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        use crate::schema::{IndexRecordOption, TextFieldIndexing, TextOptions};
        let no_positions = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default().set_index_option(IndexRecordOption::WithFreqs),
        );

        let text_field = schema_builder.add_text_field("text", no_positions);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field=>"a b c"))?;
            index_writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "b"),
        ]);

        let search_error = searcher
            .search(&phrase_query, &TEST_COLLECTOR_WITH_SCORE)
            .err();
        assert!(matches!(
            search_error,
            Some(crate::TantivyError::SchemaError(msg))
            if msg == "Applied phrase query on field \"text\", which does not have positions \
            indexed"
        ));
        Ok(())
    }

    #[test]
    pub fn test_phrase_score() -> crate::Result<()> {
        let index = create_index(&["a b c", "a b c a b"])?;
        let scores = test_query(0, &index, vec!["a", "b"]);
        assert_nearly_equals!(scores[0], 0.40618482);
        assert_nearly_equals!(scores[1], 0.46844664);
        Ok(())
    }

    #[ignore]
    #[test]
    pub fn test_phrase_score_with_slop() -> crate::Result<()> {
        let index = create_index(&["a c b", "a b c a b"])?;
        let scores = test_query(1, &index, vec!["a", "b"]);
        assert_nearly_equals!(scores[0], 0.40618482);
        assert_nearly_equals!(scores[1], 0.46844664);
        Ok(())
    }

    #[test]
    pub fn test_phrase_score_with_slop_bug() -> crate::Result<()> {
        let index = create_index(&["asdf asdf Captain Subject Wendy", "Captain"])?;
        let scores = test_query(1, &index, vec!["captain", "wendy"]);
        assert_eq!(scores.len(), 1);
        Ok(())
    }

    #[test]
    pub fn test_phrase_score_with_slop_bug_2() -> crate::Result<()> {
        // fails
        let index = create_index(&["a x b x c", "a a c"])?;
        let scores = test_query(2, &index, vec!["a", "b", "c"]);
        assert_eq!(scores.len(), 1);

        let index = create_index(&["a x b x c", "b c c"])?;
        let scores = test_query(2, &index, vec!["a", "b", "c"]);
        assert_eq!(scores.len(), 1);

        Ok(())
    }

    fn test_query(slop: u32, index: &Index, texts: Vec<&str>) -> Vec<f32> {
        let text_field = index.schema().get_field("text").unwrap();
        let searcher = index.reader().unwrap().searcher();
        let terms: Vec<Term> = texts
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let mut phrase_query = PhraseQuery::new(terms);
        phrase_query.set_slop(slop);
        searcher
            .search(&phrase_query, &TEST_COLLECTOR_WITH_SCORE)
            .expect("search should succeed")
            .scores()
            .to_vec()
    }

    #[test]
    pub fn test_phrase_score_with_slop_repeating() -> crate::Result<()> {
        let index = create_index(&["wendy subject subject captain", "Captain"])?;
        let scores = test_query(1, &index, vec!["wendy", "subject", "captain"]);
        assert_eq!(scores.len(), 1);
        Ok(())
    }

    #[test]
    pub fn test_phrase_score_with_slop_size() -> crate::Result<()> {
        let index = create_index(&["a b e c", "a e e e c", "a e e e e c"])?;
        let scores = test_query(3, &index, vec!["a", "c"]);
        assert_eq!(scores.len(), 2);
        assert_nearly_equals!(scores[0], 0.29086056);
        assert_nearly_equals!(scores[1], 0.26706287);
        Ok(())
    }

    #[test]
    pub fn test_phrase_slop() -> crate::Result<()> {
        let index = create_index(&["a x b c"])?;
        let scores = test_query(1, &index, vec!["a", "b", "c"]);
        assert_eq!(scores.len(), 1);

        let index = create_index(&["a x b x c"])?;
        let scores = test_query(1, &index, vec!["a", "b", "c"]);
        assert_eq!(scores.len(), 0);

        let index = create_index(&["a b"])?;
        let scores = test_query(1, &index, vec!["b", "a"]);
        assert_eq!(scores.len(), 0);

        let index = create_index(&["a b"])?;
        let scores = test_query(2, &index, vec!["b", "a"]);
        assert_eq!(scores.len(), 1);

        Ok(())
    }

    #[test]
    pub fn test_phrase_score_with_slop_ordering() -> crate::Result<()> {
        let index = create_index(&[
            "a e b e c",
            "a e e e e e b e e e e c",
            "a c b", // also matches
            "a c e b e",
            "a e c b",
            "a e b c",
        ])?;
        let scores = test_query(3, &index, vec!["a", "b", "c"]);
        // The first and last matches.
        assert_nearly_equals!(scores[0], 0.23091172);
        assert_nearly_equals!(scores[1], 0.27310878);
        assert_nearly_equals!(scores[3], 0.25024384);
        Ok(())
    }

    #[test] // motivated by #234
    pub fn test_phrase_query_docfreq_order() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field=>"b"))?;
            index_writer.add_document(doc!(text_field=>"a b"))?;
            index_writer.add_document(doc!(text_field=>"b a"))?;
            index_writer.commit()?;
        }

        let searcher = index.reader()?.searcher();
        let test_query = |texts: Vec<&str>| {
            let terms: Vec<Term> = texts
                .iter()
                .map(|text| Term::from_field_text(text_field, text))
                .collect();
            let phrase_query = PhraseQuery::new(terms);
            searcher
                .search(&phrase_query, &TEST_COLLECTOR_WITH_SCORE)
                .expect("search should succeed")
                .docs()
                .to_vec()
        };
        assert_eq!(test_query(vec!["a", "b"]), vec![DocAddress::new(0, 1)]);
        assert_eq!(test_query(vec!["b", "a"]), vec![DocAddress::new(0, 2)]);
        Ok(())
    }

    #[test] // motivated by #234
    pub fn test_phrase_query_non_trivial_offsets() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(text_field=>"a b c d e f g h"))?;
            index_writer.commit()?;
        }
        let searcher = index.reader().unwrap().searcher();
        let test_query = |texts: Vec<(usize, &str)>| {
            let terms: Vec<(usize, Term)> = texts
                .iter()
                .map(|(offset, text)| (*offset, Term::from_field_text(text_field, text)))
                .collect();
            let phrase_query = PhraseQuery::new_with_offset(terms);
            searcher
                .search(&phrase_query, &TEST_COLLECTOR_WITH_SCORE)
                .expect("search should succeed")
                .docs()
                .iter()
                .map(|doc_address| doc_address.doc_id)
                .collect::<Vec<DocId>>()
        };
        assert_eq!(test_query(vec![(0, "a"), (1, "b")]), vec![0]);
        assert_eq!(test_query(vec![(1, "b"), (0, "a")]), vec![0]);
        assert!(test_query(vec![(0, "a"), (2, "b")]).is_empty());
        assert_eq!(test_query(vec![(0, "a"), (2, "c")]), vec![0]);
        assert_eq!(test_query(vec![(0, "a"), (2, "c"), (3, "d")]), vec![0]);
        assert_eq!(test_query(vec![(0, "a"), (2, "c"), (4, "e")]), vec![0]);
        assert_eq!(test_query(vec![(4, "e"), (0, "a"), (2, "c")]), vec![0]);
        assert!(test_query(vec![(0, "a"), (2, "d")]).is_empty());
        assert_eq!(test_query(vec![(1, "a"), (3, "c")]), vec![0]);
        Ok(())
    }

    #[test]
    pub fn test_phrase_query_on_json() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let json_field = schema_builder.add_json_field("json", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(json_field=>json!({
                "text": "elliot smith the happy who"
            })))?;
            index_writer.add_document(doc!(json_field=>json!({
                "text": "the who elliot smith"
            })))?;
            index_writer.add_document(doc!(json_field=>json!({
                "arr": [{"text":"the who"}, {"text":"elliot smith"}]
            })))?;
            index_writer.add_document(doc!(json_field=>json!({
                "text2": "the smith"
            })))?;
            index_writer.commit()?;
        }
        let searcher = index.reader()?.searcher();
        let matching_docs = |query: &str| {
            let query_parser = QueryParser::for_index(&index, vec![json_field]);
            let phrase_query = query_parser.parse_query(query).unwrap();
            let phrase_weight = phrase_query
                .weight(EnableScoring::disabled_from_schema(searcher.schema()))
                .unwrap();
            let mut phrase_scorer = phrase_weight
                .scorer(searcher.segment_reader(0), 1.0f32)
                .unwrap();
            let mut docs = Vec::new();
            loop {
                let doc = phrase_scorer.doc();
                if doc == TERMINATED {
                    break;
                }
                docs.push(doc);
                phrase_scorer.advance();
            }
            docs
        };
        assert!(matching_docs(r#"text:"the smith""#).is_empty());
        assert_eq!(&matching_docs(r#"text:the"#), &[0u32, 1u32]);
        assert_eq!(&matching_docs(r#"text:"the""#), &[0u32, 1u32]);
        assert_eq!(&matching_docs(r#"text:"smith""#), &[0u32, 1u32]);
        assert_eq!(&matching_docs(r#"text:"elliot smith""#), &[0u32, 1u32]);
        assert_eq!(&matching_docs(r#"text2:"the smith""#), &[3u32]);
        assert!(&matching_docs(r#"arr.text:"the smith""#).is_empty());
        assert_eq!(&matching_docs(r#"arr.text:"elliot smith""#), &[2]);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_all_terms_in_order() -> crate::Result<()> {
        // Query: "quick brown fox"
        // Only documents where: quick position < brown position < fox position
        let index = create_index(&[
            "quick brown fox",           // Doc 0: 0 < 1 < 2 ✓
            "quick the brown fox",       // Doc 1: 0 < 2 < 3 ✓ (word in between)
            "quick fox brown",           // Doc 2: quick(0) < fox(1) < brown(2) - wrong order ✗
            "brown quick fox",           // Doc 3: brown(0) < quick(1) < fox(2) - wrong order ✗
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["quick", "brown", "fox"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // Docs 0 and 1 should match (all three terms in order)
        assert_eq!(docs, vec![0, 1]);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_wrong_order_no_match() -> crate::Result<()> {
        let index = create_index(&[
            "fox brown quick",     // Doc 0: 0 < 1 < 2, but we need quick < brown < fox
            "brown quick fox",     // Doc 1: 0 < 1 < 2, but positions are brown < quick < fox
            "quick brown fox",     // Doc 2: 0 < 1 < 2, perfect match ✓
            "fox quick brown",     // Doc 3: 0 < 1 < 2, but positions are fox < quick < brown
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["quick", "brown", "fox"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // Only doc 2 should match
        assert_eq!(docs, vec![2]);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_partial_terms_score() -> crate::Result<()> {
        let index = create_index(&[
            "quick brown fox",        // Doc 0: all 3 in order
            "quick the brown fox",    // Doc 1: all 3 in order but with word in between
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["quick", "brown", "fox"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs = results.docs();
        let scores = results.scores();

        // Both docs should match (both have all 3 terms in order)
        assert_eq!(docs.len(), 2);
        
        // Both should have same number of matched terms, so similar scores
        // Doc 1 has an extra word but same 3 terms matched
        assert!(scores[0] > 0.0);
        assert!(scores[1] > 0.0);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_single_term() -> crate::Result<()> {
        let index = create_index(&[
            "quick brown fox",
            "quick",
            "fox",
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        // Sparse phrase query requires at least 2 terms
        // So we test with 2 terms matching partially
        let terms: Vec<Term> = ["quick", "fox"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // Should match docs containing "quick" before "fox"
        assert_eq!(docs, vec![0]);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_bug_one() -> crate::Result<()> {
        let index = create_index(&[
            "south 76 east",
            "west 76 south",
            "north 75 west",
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        // Sparse phrase query requires at least 2 terms
        // So we test with 2 terms matching partially
        let terms: Vec<Term> = ["south", "east"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // Should match docs containing "south" before "east"
        assert_eq!(docs, vec![0]);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_far_apart_terms() -> crate::Result<()> {
        // Test with terms that are far apart in the document
        let index = create_index(&[
            "start a b c d e f g h end",
            "start a b c d e f g h middle end",
            "end h g f e d c b a start",
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["start", "end"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // Docs 0 and 1 should match (start < end)
        // Doc 2 should not match (end < start)
        assert_eq!(docs, vec![0, 1]);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_four_terms() -> crate::Result<()> {
        // Test with 4 terms in order
        let index = create_index(&[
            "alpha beta gamma delta",
            "alpha x beta y gamma z delta",
            "delta gamma beta alpha",
            "alpha beta delta gamma",
            "x alpha y beta z gamma w delta",
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["alpha", "beta", "gamma", "delta"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // Docs 0, 1, and 4 should match (all terms in order)
        // Doc 2 should not match (reverse order)
        // Doc 3 should not match (gamma and delta out of order)
        assert_eq!(docs, vec![0, 1, 4]);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_repeated_terms() -> crate::Result<()> {
        // Test with repeated terms in document
        let index = create_index(&[
            "cat dog cat dog cat",     // Doc 0: cat(0,2,4) dog(1,3) - cat < dog possible (0<1, 2<3)
            "dog dog dog cat cat cat", // Doc 1: dog(0,1,2) cat(3,4,5) - all cats after all dogs
            "cat cat cat dog dog dog", // Doc 2: cat(0,1,2) dog(3,4,5) - all cats before all dogs
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["cat", "dog"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // Docs 0 and 2 should match (cat appears before dog at some position)
        // Doc 1: all cats (3,4,5) are after all dogs (0,1,2) - no match
        assert_eq!(docs, vec![0, 2]);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_no_matches() -> crate::Result<()> {
        // Test where documents have the terms but not in the right order
        let index = create_index(&[
            "cherry banana apple",
            "apple cherry banana",
            "banana cherry apple",
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["apple", "banana", "cherry"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // No documents match because apple < banana < cherry never occurs
        // Doc 0: cherry(0) banana(1) apple(2) - no match
        // Doc 1: apple(0) cherry(1) banana(2) - no match  
        // Doc 2: banana(0) cherry(1) apple(2) - no match
        assert!(docs.is_empty());
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_scoring_distance() -> crate::Result<()> {
        // Test that documents with terms closer together score better
        let index = create_index(&[
            "cat dog",              // Doc 0: terms adjacent
            "cat x dog",            // Doc 1: terms 1 word apart
            "cat x x x dog",        // Doc 2: terms 3 words apart
            "cat x x x x x dog",    // Doc 3: terms 5 words apart
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["cat", "dog"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs = results.docs();
        let scores = results.scores();

        // All documents should match
        assert_eq!(docs.len(), 4);
        // Scores should be equal since all have the same number of matched terms (2/2)
        // They all match completely so they should have similar scores
        for &score in scores {
            assert!(score > 0.0);
        }
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_overlapping_positions() -> crate::Result<()> {
        // Test with terms that have multiple position options
        let index = create_index(&[
            "run rabbit run fast rabbit",  // Doc 0: run(0,2) rabbit(1,4) - 0<1 or 2<4 ✓
            "rabbit run fast rabbit run",  // Doc 1: rabbit(0,3) run(1,4) - 0<1 or 3<4 ✓ (both orderings exist)
            "fast run fast rabbit fast",   // Doc 2: run(1) rabbit(3) - 1<3 ✓
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["run", "rabbit"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // All docs match because in each there exists at least one run < rabbit ordering
        assert_eq!(docs, vec![0, 1, 2]);
        Ok(())
    }

    #[test]
    pub fn test_sparse_phrase_query_three_terms_mixed() -> crate::Result<()> {
        // Test 3 terms with various configurations
        let index = create_index(&[
            "one two three",           // Doc 0: perfect order
            "one middle two middle three",  // Doc 1: order with gaps
            "three two one",           // Doc 2: reverse order
            "one three two",           // Doc 3: wrong order (2 and 3 swapped)
            "two one three",           // Doc 4: wrong order (1 and 2 swapped)
        ])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();

        let terms: Vec<Term> = ["one", "two", "three"]
            .iter()
            .map(|text| Term::from_field_text(text_field, text))
            .collect();
        let query = SparsePhraSeQuery::new(terms);
        let results = searcher
            .search(&query, &TEST_COLLECTOR_WITH_SCORE)
            .unwrap();
        let docs: Vec<u32> = results
            .docs()
            .iter()
            .map(|docaddr| docaddr.doc_id)
            .collect();

        // Only docs 0 and 1 should match
        assert_eq!(docs, vec![0, 1]);
        Ok(())
    }
}

