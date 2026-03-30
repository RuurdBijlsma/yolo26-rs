use color_eyre::Result;
use object_detector::YOLO26Predictor;
use std::collections::HashSet;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let mut predictor = YOLO26Predictor::from_hf().await?;

    let image = Path::new("assets/img/market.jpg");
    let img = image::open(image)?;

    let results = predictor.predict(&img, 0.4, 0.7)?;

    let tags = results
        .iter()
        .map(|r| r.tag.clone())
        .collect::<HashSet<_>>();

    println!("{} - Objects: {:?}", image.display(), tags);
    Ok(())
}
