from fastapi import APIRouter
from .animal_route import router as animal_cls_route

router = APIRouter()
router.include_router(animal_cls_route, prefix="/animal_classification")