@Echo Off
call conda activate tensorflow

call set PYTHONHOME=C:\tools\miniconda3\envs\tensorflow

call tensorboard.exe --logdir data/logs/fit

pause