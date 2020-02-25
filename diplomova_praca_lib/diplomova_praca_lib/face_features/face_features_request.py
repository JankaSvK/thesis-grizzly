from diplomova_praca_lib.storage import FileStorage, Database

database = Database(FileStorage.load_data_from_file(r"C:\Users\janul\Desktop\saved_annotations\50-faces.npy"))

def face_features_request(request):
    return None