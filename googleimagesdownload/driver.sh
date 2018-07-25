
#Clean up ./
sudo rm -rf ./data/*

#Download image from google image search
sudo googleimagesdownload -cf ./config.json -ri --chromedriver /Users/cliu/Documents/Banner-Reflow-ML/ 
sudo chmod -R 777 data/

#Clean up
find ./data -type f  ! -name "*.jpg"  -delete
find ./data -type f  ! -name "*.*"  -delete


#Delete currupted jpg images
python verify_data_set.py


#Run Training
#python main_resnet50.py