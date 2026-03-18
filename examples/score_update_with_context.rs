// # Score Update Query With Context Example
//
// This example demonstrates how to use the `ScoreUpdateQueryWithContext` to modify
// document scores based on field values and custom context data.

use tantivy::collector::TopDocs;
use tantivy::query::{AllQuery, ScoreUpdateQueryWithContext, ScoringContext, TermQuery};
use tantivy::schema::{Schema, Value, FAST, STORED, TEXT};
use tantivy::{doc, Index, IndexWriter, TantivyDocument, Term};

fn main() -> tantivy::Result<()> {
    // Create a schema with text and numeric fields
    let mut schema_builder = Schema::builder();
    let title = schema_builder.add_text_field("title", TEXT | STORED);
    let popularity = schema_builder.add_u64_field("popularity", FAST | STORED);
    let timestamp = schema_builder.add_u64_field("timestamp", FAST | STORED);
    let schema = schema_builder.build();

    // Create an index in RAM
    let index = Index::create_in_ram(schema.clone());
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;

    // Add some documents with popularity and timestamp values
    index_writer.add_document(doc!(
        title => "Introduction to Rust",
        popularity => 150u64,
        timestamp => 1609459200u64 // 2021-01-01
    ))?;

    index_writer.add_document(doc!(
        title => "Advanced Rust Programming",
        popularity => 300u64,
        timestamp => 1640995200u64 // 2022-01-01
    ))?;

    index_writer.add_document(doc!(
        title => "Rust for Beginners",
        popularity => 500u64,
        timestamp => 1672531200u64 // 2023-01-01
    ))?;

    index_writer.add_document(doc!(
        title => "Mastering Rust",
        popularity => 200u64,
        timestamp => 1704067200u64 // 2024-01-01
    ))?;

    index_writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();

    println!("=== Example 1: Boosting by Popularity ===\n");

    // Create a term query for "rust"
    let term_query = TermQuery::new(
        Term::from_field_text(title, "rust"),
        Default::default(),
    );

    // Create scoring context with popularity field
    let popularity_context = ScoringContext::new(popularity).with_base_value(100);

    // Boost scores based on popularity relative to base value
    let popularity_boosted_query = ScoreUpdateQueryWithContext::with_fn(
        Box::new(term_query.clone()),
        popularity_context,
        |field_val, base_val, score, _order| {
            let boost = field_val.unwrap_or(0) as f32 / base_val.unwrap_or(100) as f32;
            score * (1.0 + boost)
        },
    );

    let top_docs = searcher.search(&popularity_boosted_query, &TopDocs::with_limit(10).order_by_score())?;

    println!("Documents boosted by popularity (base=100):");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let doc_title = retrieved_doc.get_first(title).unwrap().as_str().unwrap();
        let doc_popularity = retrieved_doc.get_first(popularity).unwrap().as_u64().unwrap();
        println!(
            "  Score: {:.4} - Title: {:<30} Popularity: {}",
            score, doc_title, doc_popularity
        );
    }

    println!("\n=== Example 2: Boosting by Recency (Descending Order) ===\n");

    // Boost by recency - newer timestamps get higher scores
    let recency_context = ScoringContext::new(timestamp)
        .with_base_value(1640995200) // 2022-01-01 as reference
        .with_sort_order(2); // 2 = Descending (prefer higher values)

    let recency_boosted_query = ScoreUpdateQueryWithContext::with_fn(
        Box::new(term_query.clone()),
        recency_context,
        |field_val, base_val, score, sort_order| {
            if let (Some(val), Some(base)) = (field_val, base_val) {
                match sort_order {
                    Some(2) => {
                        // Descending - boost newer documents
                        let recency_boost = (val as f32 / base as f32).max(0.5);
                        score * recency_boost
                    }
                    Some(1) => {
                        // Ascending - boost older documents
                        let boost = (base as f32 / val as f32).max(0.5);
                        score * boost
                    }
                    _ => score,
                }
            } else {
                score
            }
        },
    );

    let top_docs = searcher.search(&recency_boosted_query, &TopDocs::with_limit(10).order_by_score())?;

    println!("Documents boosted by recency (base=2022-01-01, prefer newer):");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let doc_title = retrieved_doc.get_first(title).unwrap().as_str().unwrap();
        let doc_timestamp = retrieved_doc.get_first(timestamp).unwrap().as_u64().unwrap();
        let year = 1970 + (doc_timestamp / 31536000); // Approximate year
        println!(
            "  Score: {:.4} - Title: {:<30} ~Year: {}",
            score, doc_title, year
        );
    }

    println!("\n=== Example 3: Combined Popularity and Recency ===\n");

    // First apply popularity boost
    let popularity_context = ScoringContext::new(popularity).with_base_value(200);
    
    let step1_query = ScoreUpdateQueryWithContext::with_fn(
        Box::new(term_query.clone()),
        popularity_context,
        |field_val, base_val, score, _order| {
            let pop_factor = (field_val.unwrap_or(0) as f32 / base_val.unwrap_or(200) as f32).max(0.5);
            score * pop_factor
        },
    );

    // Then apply recency boost on top
    let recency_context = ScoringContext::new(timestamp)
        .with_base_value(1672531200) // 2023-01-01
        .with_sort_order(2);

    let combined_query = ScoreUpdateQueryWithContext::with_fn(
        Box::new(step1_query),
        recency_context,
        |field_val, base_val, score, _sort_order| {
            if let (Some(val), Some(base)) = (field_val, base_val) {
                let recency_factor = (val as f32 / base as f32).max(0.3);
                score * recency_factor
            } else {
                score * 0.5 // Penalize missing timestamps
            }
        },
    );

    let top_docs = searcher.search(&combined_query, &TopDocs::with_limit(10).order_by_score())?;

    println!("Documents with combined popularity and recency boosting:");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let doc_title = retrieved_doc.get_first(title).unwrap().as_str().unwrap();
        let doc_popularity = retrieved_doc.get_first(popularity).unwrap().as_u64().unwrap();
        let doc_timestamp = retrieved_doc.get_first(timestamp).unwrap().as_u64().unwrap();
        let year = 1970 + (doc_timestamp / 31536000);
        println!(
            "  Score: {:.4} - Title: {:<30} Pop: {:3}, ~Year: {}",
            score, doc_title, doc_popularity, year
        );
    }

    println!("\n=== Example 4: Custom Decay Function ===\n");

    // Apply an exponential decay based on timestamp difference
    let decay_context = ScoringContext::new(timestamp)
        .with_base_value(1704067200); // 2024-01-01 as "now"

    let decay_query = ScoreUpdateQueryWithContext::with_fn(
        Box::new(AllQuery),
        decay_context,
        |field_val, base_val, score, _order| {
            if let (Some(val), Some(base)) = (field_val, base_val) {
                // Calculate days difference (approximate)
                let days_diff = ((base as i64 - val as i64).abs() / 86400) as f32;
                // Exponential decay: score * exp(-days / 365)
                let decay_factor = (-days_diff / 365.0).exp();
                score * (0.5 + 0.5 * decay_factor) // Between 0.5 and 1.0
            } else {
                score * 0.1 // Heavy penalty for missing data
            }
        },
    );

    let top_docs = searcher.search(&decay_query, &TopDocs::with_limit(10).order_by_score())?;

    println!("Documents with exponential time decay (reference: 2024-01-01):");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let doc_title = retrieved_doc.get_first(title).unwrap().as_str().unwrap();
        let doc_timestamp = retrieved_doc.get_first(timestamp).unwrap().as_u64().unwrap();
        let year = 1970 + (doc_timestamp / 31536000);
        println!(
            "  Score: {:.4} - Title: {:<30} ~Year: {}",
            score, doc_title, year
        );
    }

    println!("\n=== Example 5: Popularity Tiers ===\n");

    // Categorize by popularity tiers with different boosts
    let tier_context = ScoringContext::new(popularity);

    let tier_query = ScoreUpdateQueryWithContext::with_fn(
        Box::new(term_query),
        tier_context,
        |field_val, _base_val, score, _order| {
            let pop = field_val.unwrap_or(0);
            let tier_boost = if pop >= 400 {
                2.0 // Very popular
            } else if pop >= 250 {
                1.5 // Popular
            } else if pop >= 150 {
                1.2 // Moderately popular
            } else {
                1.0 // Less popular
            };
            score * tier_boost
        },
    );

    let top_docs = searcher.search(&tier_query, &TopDocs::with_limit(10).order_by_score())?;

    println!("Documents categorized by popularity tiers:");
    for (score, doc_address) in top_docs {
        let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
        let doc_title = retrieved_doc.get_first(title).unwrap().as_str().unwrap();
        let doc_popularity = retrieved_doc.get_first(popularity).unwrap().as_u64().unwrap();
        let tier = if doc_popularity >= 400 {
            "Very Popular"
        } else if doc_popularity >= 250 {
            "Popular"
        } else if doc_popularity >= 150 {
            "Moderate"
        } else {
            "Less Popular"
        };
        println!(
            "  Score: {:.4} - Title: {:<30} Pop: {:3} ({:12})",
            score, doc_title, doc_popularity, tier
        );
    }

    Ok(())
}
