//! PNG export/import with embedded view state ("bookmark").
//!
//! The complete view state is stored as JSON in an iTXt chunk, so any exported
//! image can be reopened to exactly the view that produced it.

use std::path::Path;

/// iTXt keyword under which the bookmark JSON is stored.
pub const BOOKMARK_KEYWORD: &str = "fractalx-bookmark";

/// Keyword used by exports from before the rename; still accepted on load.
const LEGACY_BOOKMARK_KEYWORD: &str = "selfsame-bookmark";

pub fn save_png(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &[u8],
    bookmark_json: &str,
) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .add_itxt_chunk(BOOKMARK_KEYWORD.to_owned(), bookmark_json.to_owned())
        .map_err(|e| e.to_string())?;
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgba).map_err(|e| e.to_string())?;
    Ok(())
}

/// Extract the bookmark JSON from a PNG previously written by [`save_png`].
pub fn load_bookmark_json(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let reader = decoder.read_info().map_err(|e| e.to_string())?;
    reader
        .info()
        .utf8_text
        .iter()
        .find(|c| c.keyword == BOOKMARK_KEYWORD || c.keyword == LEGACY_BOOKMARK_KEYWORD)
        .ok_or_else(|| "no FractalX bookmark found in this PNG".to_owned())?
        .get_text()
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bookmark_survives_png_round_trip() {
        let dir = std::env::temp_dir().join("fractalx-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.png");

        let json = r#"{"app":"fractalx","version":1,"view":{"center":[-0.5,0.25]}}"#;
        let rgba = vec![128u8; 8 * 8 * 4];
        save_png(&path, 8, 8, &rgba, json).unwrap();

        assert_eq!(load_bookmark_json(&path).unwrap(), json);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn png_without_bookmark_reports_error() {
        let dir = std::env::temp_dir().join("fractalx-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plain.png");

        let file = std::fs::File::create(&path).unwrap();
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), 4, 4);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[0u8; 4 * 4 * 4]).unwrap();
        drop(writer);

        assert!(load_bookmark_json(&path).is_err());

        std::fs::remove_file(&path).ok();
    }
}
