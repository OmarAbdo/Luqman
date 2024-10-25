This is a basic Flask server
============================

Step 1
------
```
.\venv\Scripts\activate
```

Step 2.1 (If you want to run the unit tests)
--------------------------------------------
```
python -m unittest discover -s app/ml/tests -p "test_*.py"
```

Step 2 (only for the first time)
-------------------------------
```
pip install -r requirements.txt
```

Step 3
------
```
python app.py
```

Then your application should be accessible at port 5000.


