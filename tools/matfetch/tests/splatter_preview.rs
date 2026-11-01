// Writes a contact sheet of every splatter style when ZESTSURFACE_SPLATTER_DUMP is set.
#[test]
fn dump_splatter_sheet() {
    let Ok(out) = std::env::var("ZESTSURFACE_SPLATTER_DUMP") else { return };
    let styles = matfetch::splatter::Style::ALL;
    let s = 256u32;
    let mut sheet = image::GrayImage::new(s * styles.len() as u32, s * 2);
    for (i, st) in styles.iter().enumerate() {
        for seed in 0..2u64 {
            let (op, _) = matfetch::splatter::render(*st, seed + 1, 1.0, s);
            image::imageops::replace(&mut sheet, &op, (i as u32 * s) as i64, (seed as u32 * s) as i64);
        }
    }
    sheet.save(out).unwrap();
}
