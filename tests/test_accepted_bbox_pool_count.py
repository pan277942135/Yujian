from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Batch, BatchCropReview, ImageAsset
from app.platform.services.crop_dataset import accepted_bbox_pool_count


def test_accepted_bbox_pool_count_requires_approved_full_scene_asset(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'accepted-pool-count.sqlite'}")
    Base = Batch.metadata
    Base.create_all(engine, tables=[Batch.__table__, ImageAsset.__table__, BatchCropReview.__table__])
    Session = sessionmaker(bind=engine)
    with Session() as db:
        db.add(Batch(batch_id="batch-1", source="fixture", manifest_uri="", raw_uri=""))
        db.flush()
        assets = [
            ImageAsset(
                batch_id="batch-1",
                image_id=f"image-{index}",
                file_name=f"image-{index}.jpg",
                object_name=f"image-{index}.jpg",
                gcs_uri=f"gs://fixture/image-{index}.jpg",
                review_status=status,
                training_eligible=True,
            )
            for index, status in enumerate(("approved", "pending"))
        ]
        db.add_all(assets)
        db.flush()
        db.add_all(
            BatchCropReview(
                batch_id="batch-1",
                image_asset_id=asset.id,
                image_id=asset.image_id,
                accepted_bbox_json="[0.1,0.1,0.4,0.4]",
                status="ACCEPTED",
            )
            for asset in assets
        )
        db.commit()

        assert accepted_bbox_pool_count(db) == 1
