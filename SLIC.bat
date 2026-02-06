@echo off
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate slic
cd /d C:\Users\Murilo_Galvao\slic
streamlit run Home.py
