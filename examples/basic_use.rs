use color_eyre::Result;
use object_detector::ObjectDetector;
use std::collections::HashSet;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    let image = Path::new("assets/img/market.jpg");
    let img = image::open(image)?;

    let mut predictor = ObjectDetector::from_hf().build().await?;
    let results = predictor.predict(&img).call()?;

    let tags = results
        .iter()
        .map(|r| r.tag.clone())
        .collect::<HashSet<_>>();
    println!("{} - Objects: {:?}", image.display(), tags);

    Ok(())
}
