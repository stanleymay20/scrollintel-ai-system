@echo off
echo Starting PostgreSQL password reset...
cd /d "C:\Program Files\PostgreSQL\17\bin"
echo ALTER USER postgres PASSWORD 'scrollintel123'; | postgres --single -D "C:\Program Files\PostgreSQL\17\data" postgres
echo Password reset complete.
pause
