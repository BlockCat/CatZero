@Echo Off
call conda activate tensorflow

call set PYTHONHOME=C:\tools\miniconda3\envs\tensorflow

call cargo run

pause