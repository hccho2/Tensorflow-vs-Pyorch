
import os,glob,shutil

def move_files_to_subdirectory():
    # 하나의 디렉토리에 모여 있는 파일을 각각의 sub directory로 이동
    category =['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 
               'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 
               'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 
               'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 
               'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 
               'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

    
    base_dir = r'D:\hccho\CommonDataset\Pet-Dataset-Oxford\images'  
    
    # 디렉토리 만들기
    for c in category:
        dir_name = os.path.join(base_dir,c)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    
    all_files = glob.glob(os.path.join(base_dir,"*.jpg"))
    
    print("파일갯수: ", len(all_files))
    
    # file move
    for f in all_files:
        basename = os.path.basename(f)
        c = '_'.join(basename.split('_')[:-1])   # basset_hound_103.jpg --> ['basset', 'hound', '103.jpg']
        if os.path.exists(os.path.join(base_dir,c)):
            shutil.move(f,os.path.join(base_dir,c))
