import os
import shutil
import random
from PIL import Image, ImageOps

class DatasetHandler:

    def __init__(self, dataset_path):
        self.train_images_path = dataset_path+'train/images'
        self.val_images_path = dataset_path+'val/images'
        self.train_labels_path = dataset_path+'train/labels'
        self.val_labels_path = dataset_path+'val/labels'

    def rename_new_files(self, path, starting_at = 1):
        files = os.listdir(path)
        
        for file in files:  
            _, extension = os.path.splitext(file)
            new_name = f'{starting_at}{extension}'
            new_path = os.path.join(path, new_name)
            actual_path = os.path.join(path, file)
            os.rename(actual_path, new_path)
            starting_at += 1
  

    def rename_all_dataset(self, root_dir):
        print(os.walk(root_dir))
        for object in os.walk(root_dir):
            print(object)


    def append_dataset(self, original_dataset_path, new_dataset_path, change_label=False, old_label='', new_label=''):
        '''
        Melhoria de qualidade de vida do código :
        Bolar lógica para descobrir qual pasta é a de treino e qual é a de validação automaticamente, baseado na quantidade de arquivos em cada uma.
        '''
        original_train_images_path = original_dataset_path+'train/images'
        original_val_images_path = original_dataset_path+'val/images'
        original_train_labels_path = original_dataset_path+'train/labels'
        original_val_labels_path = original_dataset_path+'val/labels'

        new_train_images_path = new_dataset_path+'train/images'
        new_val_images_path = new_dataset_path+'val/images'
        new_train_labels_path = new_dataset_path+'train/labels'
        new_val_labels_path = new_dataset_path+'val/labels'

        try:
            self.rename_new_files(new_train_images_path, len(os.listdir(original_train_images_path))+1)
        except:
            pass

        try:
            self.rename_new_files(new_train_labels_path, len(os.listdir(original_train_labels_path))+1)
        except:
            pass

        try:
            self.rename_new_files(new_val_images_path, len(os.listdir(original_val_images_path))+1)
        except:
            pass

        try:
            self.rename_new_files(new_val_labels_path, len(os.listdir(original_val_labels_path))+1)
        except:
            pass

        try:
            self.move_files(new_train_images_path, original_train_images_path)
        except:
            pass

        try:
            self.move_files(new_val_images_path, original_val_images_path)
        except:
            pass

        try:
            self.move_files(new_train_labels_path, original_train_labels_path)
        except:
            pass

        try:
            self.move_files(new_val_labels_path, original_val_labels_path)
        except:
            pass

        if change_label:
            self.change_labels(old_label, new_label)
    
    def change_labels(self, labels_path, old_label, new_label):
        files = os.listdir(labels_path)
        for file in files:
            with open(labels_path+file, 'r') as f:
                lines = f.readlines()
            with open(labels_path+file, 'w') as f:
                for line in lines:
                    if line[0] == old_label:
                        f.write(new_label+line[1:])
                    else:
                        f.write(line)

    def delete_lines(self, labels_path, old_label):
        files = os.listdir(labels_path)
        for file in files:
            with open(labels_path+file, 'r') as f:
                lines = f.readlines()
            with open(labels_path+file, 'w') as f:
                for line in lines:
                    if line[0] != old_label:
                        f.write(line)

    def copy_files(self, origem, destino):
        arquivos = os.listdir(origem)
        for arquivo in arquivos:
            caminho_origem = os.path.join(origem, arquivo)
            caminho_destino = os.path.join(destino, arquivo)
            shutil.copy2(caminho_origem, caminho_destino)
            
    def move_files(self, origem, destino):
        arquivos = os.listdir(origem)
        for arquivo in arquivos:
            caminho_origem = os.path.join(origem, arquivo)
            caminho_destino = os.path.join(destino, arquivo)
            shutil.move(caminho_origem, caminho_destino)

# datasetHandler = DatasetHandler('dataset1')
# #datasetHandler.rename_new_files('dataset2/train/labels/')
# datasetHandler.rename_all_dataset('dataset2/')



def rgb_to_black_and_white(input_folder, porcentagem=0.3):

    files = os.listdir(input_folder)
    num_images_to_convert = int(len(files) * porcentagem)
    images_to_convert = random.sample(files, num_images_to_convert)

    for file in files:
        print(file)
        input_path = os.path.join(input_folder, file)
        print(input_path)
        # Verifica se é um arquivo de imagem
        if os.path.isfile(input_path):
            # Abre a imagem
            img = Image.open(input_path)

            # Converte para preto e branco se for uma das selecionadas
            if file in images_to_convert:
                img = img.convert('L')  # 'L' para modo de escala de cinza

            # Salva a imagem no diretório de saída
            img.save(input_path)

            # Fecha a imagem
            img.close()

def restore_color(input_folder):

    # Lista todos os arquivos no diretório de entrada
    files = os.listdir(input_folder)

    # Loop através de todos os arquivos na pasta de entrada
    for file in files:

        input_path = os.path.join(input_folder, file)

        # Verifica se é um arquivo de imagem
        if os.path.isfile(input_path):
            # Abre a imagem
            img = Image.open(input_path)

            # Verifica se a imagem está em preto e branco (escala de cinza)
            if img.mode == 'L':
                # Converte de volta para colorido
                img = img.convert('RGB')

            # Salva a imagem no diretório de saída
            img.save(input_path)

            # Fecha a imagem
            img.close()


rgb_to_black_and_white('C:/Users/pyyyt/iCloudDrive/Codes/yolov8-streamlit-detection-tracking/YOLOv8StreamlitServer/assets/asdasf', porcentagem=0.3)
# img = Image.open('C:/Users/pyyyt/iCloudDrive/Codes/yolov8-streamlit-detection-tracking/YOLOv8StreamlitServer/assets/asdasf/detected.jpg')
# # img.convert('L')
# img = ImageOps.grayscale(img)
# img.save('C:/Users/pyyyt/iCloudDrive/Codes/yolov8-streamlit-detection-tracking/YOLOv8StreamlitServer/assets/asdasf/detected1.jpg')   

