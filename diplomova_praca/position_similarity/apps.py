from django.apps import AppConfig

from diplomova_praca_lib.position_similarity.position_similarity_request import Environment


class PositionSimilarityConfig(AppConfig):
    name = 'position_similarity'
    def ready(self):
        db_regions = r"C:\Users\janul\Desktop\saved_annotations\5videos-resnet50"
        # db_regions = r"/mnt/c/Users/janul/Desktop/saved_annotations/5videos-resnet50"

        # db_regions = r"C:\Users\janul\Desktop\saved_annotations\750_resnet50_new"
        db_spatially = r"C:\Users\janul\Desktop\saved_annotations\5videos-resnet50antepenultimate"
        # db_spatially = r"/mnt/c/Users/janul/Desktop/saved_annotations/5videos-resnet50antepenultimate"
        # db_spatially  = r"C:\Users\janul\Desktop\saved_annotations\750_resnet50antepenultima_second_try"

        Environment.initialize(regions_path=db_regions, spatially_path=db_spatially)