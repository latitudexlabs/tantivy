// # Score Update Query Example
//
// This example demonstrates how to use the `ScoreUpdateQuery` to modify
// document scores using custom functions.

use std::sync::Arc;

use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, ScoreUpdateQuery, TermQuery};
use tantivy::schema::{Schema, Value, STORED, TEXT};
use tantivy::{doc, Index, IndexWriter, TantivyDocument, Term};

fn main() -> tantivy::Result<()> {
    // Create a schema with a text field
    let mut schema_builder = Schema::builder();
    let title = schema_builder.add_text_field("title", TEXT | STORED);
    let body = schema_builder.add_text_field("body", TEXT | STORED);
    let schema = schema_builder.build();

    // Create an index in RAM
    let index = Index::create_in_ram(schema.clone());

    // Create an index writer
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;

    // Add some documents
    index_writer.add_document(doc!(
        title => "The Old Man and the Sea",
        body => "He was an old man who fished alone in a skiff in the Gulf Stream."
    ))?;

    index_writer.add_document(doc!(
        title => "Of Mice and Men",
        body => "A few miles south of Soledad, the Salinas River drops in close to the hillside."
    ))?;

    index_writer.add_document(doc!(
        title => "Old Yeller",
        body => "We called him Old Yeller. The name had a double meaning."
    ))?;

    index_writer.add_document(doc!(
        title => "The Old Man",
        body => "This is a story about an old man in the village."
    ))?;

    index_writer.commit()?;

    // Create a reader and searcher
    let reader = index.reader()?;
    let searcher = reader.searcher();

    println!("=== Example 1: Doubling Scores ===\n");

    // Create a simple term query for "old"
    let term_query = TermQuery::new(
        Term::from_field_text(title, "old"),
        Default::default(),
    );

    // Wrap it with ScoreUpdateQuery to double all scores
    let doubled_query = ScoreUpdateQuery::new(
        Box::new(term_query.clone()),
        Some(Arc::new(|score| score * 2.0)),
    );

    let top_docs = searcher.search(&doubled_query, &TopDocs::with_limit(5).order_by_score())?;

    println!("Top documents with doubled scores:");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        println!(
            "  Score: {:.4} - Title: {}",
            score,
            retrieved_doc.get_first(title).unwrap().as_str().unwrap()
        );
    }

    println!("\n=== Example 2: Logarithmic Score Transformation ===\n");

    // Apply a logarithmic transformation to reduce the gap between high and low scores
    let log_query = ScoreUpdateQuery::with_fn(
        Box::new(term_query.clone()),
        |score| (1.0 + score).ln(),
    );

    let top_docs = searcher.search(&log_query, &TopDocs::with_limit(5).order_by_score())?;

    println!("Top documents with logarithmic scores:");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        println!(
            "  Score: {:.4} - Title: {}",
            score,
            retrieved_doc.get_first(title).unwrap().as_str().unwrap()
        );
    }

    println!("\n=== Example 3: Score Clamping ===\n");

    // Clamp all scores to a maximum of 0.5
    let clamped_query = ScoreUpdateQuery::with_fn(
        Box::new(term_query.clone()),
        |score| score.min(0.5),
    );

    let top_docs = searcher.search(&clamped_query, &TopDocs::with_limit(5).order_by_score())?;

    println!("Top documents with clamped scores (max 0.5):");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        println!(
            "  Score: {:.4} - Title: {}",
            score,
            retrieved_doc.get_first(title).unwrap().as_str().unwrap()
        );
    }

    println!("\n=== Example 4: Sigmoid Score Normalization ===\n");

    // Apply sigmoid function to normalize scores between 0 and 1
    let sigmoid_query = ScoreUpdateQuery::with_fn(
        Box::new(term_query.clone()),
        |score| 1.0 / (1.0 + (-score).exp()),
    );

    let top_docs = searcher.search(&sigmoid_query, &TopDocs::with_limit(5).order_by_score())?;

    println!("Top documents with sigmoid-normalized scores:");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        println!(
            "  Score: {:.4} - Title: {}",
            score,
            retrieved_doc.get_first(title).unwrap().as_str().unwrap()
        );
    }

    println!("\n=== Example 5: Combining with Boolean Query ===\n");

    // Create a boolean query with multiple terms
    let old_query = TermQuery::new(
        Term::from_field_text(title, "old"),
        Default::default(),
    );
    let man_query = TermQuery::new(
        Term::from_field_text(title, "man"),
        Default::default(),
    );

    let boolean_query = BooleanQuery::new(vec![
        (Occur::Should, Box::new(old_query) as Box<dyn Query>),
        (Occur::Should, Box::new(man_query) as Box<dyn Query>),
    ]);

    // Wrap the boolean query with score update to boost combined scores
    let boosted_boolean = ScoreUpdateQuery::with_fn(
        Box::new(boolean_query),
        |score| score * 1.5,
    );

    let top_docs = searcher.search(&boosted_boolean, &TopDocs::with_limit(5).order_by_score())?;

    println!("Top documents from boosted boolean query:");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        println!(
            "  Score: {:.4} - Title: {}",
            score,
            retrieved_doc.get_first(title).unwrap().as_str().unwrap()
        );
    }

    println!("\n=== Example 6: Score Squaring for Emphasizing High Scores ===\n");

    // Square scores to emphasize differences between high-scoring documents
    let squared_query = ScoreUpdateQuery::with_fn(
        Box::new(term_query),
        |score| score * score,
    );

    let top_docs = searcher.search(&squared_query, &TopDocs::with_limit(5).order_by_score())?;

    println!("Top documents with squared scores:");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        println!(
            "  Score: {:.4} - Title: {}",
            score,
            retrieved_doc.get_first(title).unwrap().as_str().unwrap()
        );
    }

    Ok(())
}
