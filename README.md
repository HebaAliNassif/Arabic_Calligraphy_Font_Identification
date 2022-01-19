# Arabic_Calligraphy_Font_Identification

An Arabic Font Identification System that is be able to identify the font for a given Arabic text snippet. 

The font identification system is very useful in a wide range of applications, such as the fields of graphic design like in illustrator arts and optical character recognition for digitizing hardcover documents.Not limited to that, but it also raised interest in the analysis of historical texts authen-tic manuscript verification where we would like to know the origin of a manuscript.

We had worked with the (ACdb) Arabic Calligraphy Database containing 9 categories of computer printed Arabic text snippets.
https://drive.google.com/file/d/1dC7pwzT_RHL9B42H8-Nzf5Sant-86NV6/view

## Project Pipeline

### Training Phase
![image](https://user-images.githubusercontent.com/49316071/150108401-8cadddc5-19be-4fa4-8542-bb4730852aae.png)

### Testing Phase
![image](https://user-images.githubusercontent.com/49316071/150108557-1da0da03-37c8-4796-a63a-340fb20f4fb3.png)

## To run the project, write in cmd:
	python predict.py PathOfTestsetFolder PathOfOutputFolder
	python evaluate.py

## Libraries versions ##
jupyter_client            7.0.6
jupyter_core              4.8.1
jupyterlab_pygments       0.1.2

matplotlib                3.4.3
matplotlib-base           3.4.3
matplotlib-inline         0.1.3

notebook                  6.4.5
numpy                     1.21.3
olefile                   0.46
opencv                    4.5.3
openjpeg                  2.4.0
openssl                   1.1.1l
packaging                 21.0
pandas                    1.3.4
pandoc                    2.14.2
pandocfilters             1.5.0

pickleshare               0.7.5
pillow                    8.3.2
pip                       21.3

py-opencv                 4.5.3

python                    3.9.7
python-dateutil           2.8.2
python_abi                3.9

scikit-image              0.16.2
scikit-learn              1.0.1
scipy                     1.7.1
