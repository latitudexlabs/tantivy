//! Fused geo query for indexes sorted by a Morton (Z-order) code.
//!
//! `SortedFieldGeoBoxQuery` retrieves the documents whose coordinates fall
//! inside a geographic bounding box and scores them by proximity to an origin
//! point — in a single pass:
//!
//! 1. The index is sorted ascending by a `u64` fast field holding the Morton
//!    code, so the docs whose code lies between the box corners form one
//!    contiguous doc-id run per segment, found by binary search.
//! 2. Each doc in the run is checked against the exact `f64` lat/lon columns
//!    (this removes the Z-order excursions between the corners).
//! 3. Survivors are scored inline from the same coordinates:
//!    `1 + 1/d²` (squared equirectangular degrees, clamped), so nearer docs
//!    score higher. This mirrors the generic distance-scoring callback but
//!    skips re-reading and decoding the Morton code.
//!
//! Compared to composing `SortedBoolean` + range queries, this skips the
//! intersection machinery and per-doc virtual dispatch entirely.

use std::ops::RangeInclusive;
use std::sync::Arc;

use columnar::{Cardinality, ColumnValues};

use crate::index::Order;
use crate::query::range_query::range_query_fastfield::partition_point_docs;
use crate::query::{EnableScoring, Explanation, Query, Scorer, Weight};
use crate::schema::Field;
use crate::{DocId, DocSet, Score, SegmentReader, TantivyError, TERMINATED};

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;
/// Clamp for the squared-degree distance: ~1e-6 deg (~10cm), far below
/// address precision, avoids division by zero for the exact origin point.
const MIN_SQUARED_DISTANCE: f64 = 1e-12;

/// See the module documentation.
#[derive(Clone, Debug)]
pub struct SortedFieldGeoBoxQuery {
    sort_field: Field,
    sort_value_range: RangeInclusive<u64>,
    lat_field: Field,
    lon_field: Field,
    lat_range: RangeInclusive<f64>,
    lon_range: RangeInclusive<f64>,
    origin_lat: f64,
    origin_lon: f64,
}

impl SortedFieldGeoBoxQuery {
    /// Creates the query.
    ///
    /// * `sort_field` — the `u64` fast field the index is sorted by
    ///   (ascending); validated at weight creation.
    /// * `sort_value_range` — the sort-code interval covering the box (e.g.
    ///   Morton codes of the SW and NE corners).
    /// * `lat_field`/`lon_field` — full-cardinality `f64` fast fields.
    /// * `lat_range`/`lon_range` — the exact bounding box.
    /// * `origin_*` — the point distances are scored against.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sort_field: Field,
        sort_value_range: RangeInclusive<u64>,
        lat_field: Field,
        lon_field: Field,
        lat_range: RangeInclusive<f64>,
        lon_range: RangeInclusive<f64>,
        origin_lat: f64,
        origin_lon: f64,
    ) -> SortedFieldGeoBoxQuery {
        SortedFieldGeoBoxQuery {
            sort_field,
            sort_value_range,
            lat_field,
            lon_field,
            lat_range,
            lon_range,
            origin_lat,
            origin_lon,
        }
    }
}

impl Query for SortedFieldGeoBoxQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let schema = enable_scoring.schema();
        let sort_field_name = schema.get_field_entry(self.sort_field).name().to_string();

        // The contiguous-run retrieval is only correct when the index is
        // sorted ascending by exactly this field — fail loudly otherwise.
        let searcher = match &enable_scoring {
            EnableScoring::Enabled { searcher, .. } => Some(*searcher),
            EnableScoring::Disabled { searcher_opt, .. } => *searcher_opt,
        };
        let sorted_asc = searcher
            .and_then(|searcher| searcher.index().settings().sort_by_field.as_ref())
            .map(|sort| sort.field == sort_field_name && sort.order == Order::Asc)
            .unwrap_or(false);
        if !sorted_asc {
            return Err(TantivyError::SchemaError(format!(
                "SortedFieldGeoBoxQuery requires the index to be sorted ascending by \
                 field {sort_field_name:?}"
            )));
        }

        Ok(Box::new(SortedFieldGeoBoxWeight {
            query: self.clone(),
            sort_field_name,
            lat_field_name: schema.get_field_entry(self.lat_field).name().to_string(),
            lon_field_name: schema.get_field_entry(self.lon_field).name().to_string(),
        }))
    }
}

struct SortedFieldGeoBoxWeight {
    query: SortedFieldGeoBoxQuery,
    sort_field_name: String,
    lat_field_name: String,
    lon_field_name: String,
}

impl Weight for SortedFieldGeoBoxWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let sort_column = reader.fast_fields().u64(&self.sort_field_name)?;
        if sort_column.index.get_cardinality() != Cardinality::Full {
            return Err(TantivyError::SchemaError(format!(
                "SortedFieldGeoBoxQuery requires a full-cardinality sort column, \
                 field {:?}",
                self.sort_field_name
            )));
        }
        let lat_column = reader.fast_fields().f64(&self.lat_field_name)?;
        let lon_column = reader.fast_fields().f64(&self.lon_field_name)?;
        if lat_column.index.get_cardinality() != Cardinality::Full
            || lon_column.index.get_cardinality() != Cardinality::Full
        {
            return Err(TantivyError::SchemaError(format!(
                "SortedFieldGeoBoxQuery requires full-cardinality lat/lon columns \
                 ({:?}, {:?})",
                self.lat_field_name, self.lon_field_name
            )));
        }

        // Contiguous doc-id run holding all sort values in range.
        let num_docs = sort_column.num_docs();
        let values = &sort_column.values;
        let range = &self.query.sort_value_range;
        let start = partition_point_docs(num_docs, |doc| values.get_val(doc) < *range.start());
        let end = partition_point_docs(num_docs, |doc| values.get_val(doc) <= *range.end());

        let mut scorer = SortedFieldGeoBoxScorer {
            doc: if start >= end { TERMINATED } else { start },
            end,
            lat_values: Arc::clone(&lat_column.values),
            lon_values: Arc::clone(&lon_column.values),
            lat_range: self.query.lat_range.clone(),
            lon_range: self.query.lon_range.clone(),
            origin_lat: self.query.origin_lat,
            origin_lon: self.query.origin_lon,
            score: 0.0,
            boost,
        };
        // Position on the first matching doc (DocSet convention).
        if scorer.doc != TERMINATED && !scorer.settle_on_match() {
            scorer.doc = TERMINATED;
        }
        Ok(Box::new(scorer))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(format!(
                "Document #({doc}) does not match"
            )));
        }
        Ok(Explanation::new("SortedFieldGeoBox", scorer.score()))
    }
}

struct SortedFieldGeoBoxScorer {
    doc: DocId,
    end: DocId,
    lat_values: Arc<dyn ColumnValues<f64>>,
    lon_values: Arc<dyn ColumnValues<f64>>,
    lat_range: RangeInclusive<f64>,
    lon_range: RangeInclusive<f64>,
    origin_lat: f64,
    origin_lon: f64,
    score: Score,
    boost: Score,
}

impl SortedFieldGeoBoxScorer {
    /// If the current doc is inside the box, computes its score and returns
    /// true. Otherwise advances until a matching doc (returning true) or the
    /// end of the run (returning false, doc left at `end`).
    #[inline]
    fn settle_on_match(&mut self) -> bool {
        while self.doc < self.end {
            let lat = self.lat_values.get_val(self.doc);
            if *self.lat_range.start() <= lat && lat <= *self.lat_range.end() {
                let lon = self.lon_values.get_val(self.doc);
                if *self.lon_range.start() <= lon && lon <= *self.lon_range.end() {
                    // Squared equirectangular distance in degrees — mirrors
                    // commons::geo_search::simplified_distance.
                    let x = (lon - self.origin_lon)
                        * f64::cos(DEG2RAD * (lat + self.origin_lat) / 2.0);
                    let y = lat - self.origin_lat;
                    let squared_distance = (x * x + y * y).max(MIN_SQUARED_DISTANCE);
                    // Nearer -> higher, same shape as the generic distance
                    // scoring callback (score + score/distance with score=1).
                    self.score = self.boost * (1.0 + 1.0 / squared_distance as Score);
                    return true;
                }
            }
            self.doc += 1;
        }
        false
    }
}

impl DocSet for SortedFieldGeoBoxScorer {
    fn advance(&mut self) -> DocId {
        if self.doc == TERMINATED {
            return TERMINATED;
        }
        self.doc += 1;
        if !self.settle_on_match() {
            self.doc = TERMINATED;
        }
        self.doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.doc == TERMINATED {
            return TERMINATED;
        }
        if target > self.doc {
            self.doc = target;
            if !self.settle_on_match() {
                self.doc = TERMINATED;
            }
        }
        self.doc
    }

    fn doc(&self) -> DocId {
        self.doc
    }

    fn size_hint(&self) -> u32 {
        // Upper bound: the remaining run length.
        if self.doc == TERMINATED {
            0
        } else {
            self.end - self.doc
        }
    }
}

impl Scorer for SortedFieldGeoBoxScorer {
    fn score(&mut self) -> Score {
        self.score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::TopDocs;
    use crate::index::{IndexSortByField, Order};
    use crate::schema::{Schema, FAST, INDEXED};
    use crate::{doc, Index, IndexSettings};

    fn interleave(x: u64) -> u64 {
        let mut x = x & 0xFFFF_FFFF;
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333;
        x = (x | (x << 1)) & 0x5555_5555_5555_5555;
        x
    }
    fn morton(lat: f64, lon: f64) -> u64 {
        let x = ((lat + 90.0) * 1e7) as u64;
        let y = ((lon + 180.0) * 1e7) as u64;
        (interleave(y) << 1) | interleave(x)
    }
    fn squared_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let x = (lon2 - lon1) * f64::cos(DEG2RAD * (lat1 + lat2) / 2.0);
        let y = lat2 - lat1;
        x * x + y * y
    }

    fn build_geo_index(points: &[(f64, f64)], sorted: bool) -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let morton_field = schema_builder.add_u64_field("morton_code", FAST | INDEXED);
        let lat_field = schema_builder.add_f64_field("lat_ff", FAST | INDEXED);
        let lon_field = schema_builder.add_f64_field("lon_ff", FAST | INDEXED);
        let schema = schema_builder.build();
        let mut builder = Index::builder().schema(schema);
        if sorted {
            builder = builder.settings(IndexSettings {
                sort_by_field: Some(IndexSortByField {
                    field: "morton_code".to_string(),
                    order: Order::Asc,
                }),
                ..IndexSettings::default()
            });
        }
        let index = builder.create_in_ram()?;
        let mut writer = index.writer_for_tests()?;
        for (i, (lat, lon)) in points.iter().enumerate() {
            writer.add_document(doc!(
                morton_field => morton(*lat, *lon),
                lat_field => *lat,
                lon_field => *lon,
            ))?;
            if i % 40 == 39 {
                writer.commit()?; // several segments
            }
        }
        writer.commit()?;
        Ok(index)
    }

    fn make_query(index: &Index, origin: (f64, f64), delta: f64) -> SortedFieldGeoBoxQuery {
        let schema = index.schema();
        let (lat, lon) = origin;
        let (lat_range, lon_range) = (lat - delta..=lat + delta, lon - delta..=lon + delta);
        SortedFieldGeoBoxQuery::new(
            schema.get_field("morton_code").unwrap(),
            morton(*lat_range.start(), *lon_range.start())
                ..=morton(*lat_range.end(), *lon_range.end()),
            schema.get_field("lat_ff").unwrap(),
            schema.get_field("lon_ff").unwrap(),
            lat_range,
            lon_range,
            lat,
            lon,
        )
    }

    #[test]
    fn geo_box_query_returns_exactly_box_members_nearest_first() -> crate::Result<()> {
        // A grid of points around an origin, plus far-away noise.
        let origin = (38.99, -76.91);
        let mut points = Vec::new();
        for i in -10i32..=10 {
            for j in -10i32..=10 {
                points.push((
                    origin.0 + (i as f64) * 0.001,
                    origin.1 + (j as f64) * 0.001,
                ));
            }
        }
        points.push((51.5, -0.12)); // London
        points.push((-33.86, 151.2)); // Sydney
        let delta = 0.0035; // captures a 7x7 sub-grid

        let index = build_geo_index(&points, true)?;
        let query = make_query(&index, origin, delta);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let top = searcher.search(&query, &TopDocs::with_limit(1000).order_by_score())?;

        // Expected: exactly the points within the box.
        let expected_count = points
            .iter()
            .filter(|(la, lo)| {
                (la - origin.0).abs() <= delta + 1e-12 && (lo - origin.1).abs() <= delta + 1e-12
            })
            .count();
        assert_eq!(top.len(), expected_count, "box membership mismatch");

        // Ordering: scores descending == true distance ascending.
        let mut last_distance = -1.0f64;
        for (_score, addr) in &top {
            let seg = searcher.segment_reader(addr.segment_ord);
            let lat = seg.fast_fields().f64("lat_ff")?.values.get_val(addr.doc_id);
            let lon = seg.fast_fields().f64("lon_ff")?.values.get_val(addr.doc_id);
            let d = squared_distance(origin.0, origin.1, lat, lon);
            assert!(
                d + 1e-15 >= last_distance,
                "results not in nearest-first order: {d} after {last_distance}"
            );
            last_distance = d;
        }
        Ok(())
    }

    #[test]
    fn geo_box_query_empty_when_nothing_in_box() -> crate::Result<()> {
        let points = vec![(10.0, 10.0), (11.0, 11.0)];
        let index = build_geo_index(&points, true)?;
        let query = make_query(&index, (45.0, 45.0), 0.01);
        let reader = index.reader()?;
        let top = reader.searcher().search(&query, &TopDocs::with_limit(10).order_by_score())?;
        assert!(top.is_empty());
        Ok(())
    }

    #[test]
    fn geo_box_query_rejects_unsorted_index() -> crate::Result<()> {
        let points = vec![(10.0, 10.0), (11.0, 11.0)];
        let index = build_geo_index(&points, false)?;
        let query = make_query(&index, (10.0, 10.0), 0.5);
        let reader = index.reader()?;
        let result = reader.searcher().search(&query, &TopDocs::with_limit(10).order_by_score());
        assert!(result.is_err(), "unsorted index must be rejected");
        Ok(())
    }
}
