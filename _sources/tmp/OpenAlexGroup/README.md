# OpenAlexGroup
# Fast API
See install options and usage at: https://fastapi.tiangolo.com/
```
$ uvicorn main:app --reload --host
```
and copy and paste the url in your browser, e.g:

http://127.0.0.1:8000/

You can now filter with the JSON keys into the url, e.g:

http://127.0.0.1:8000/?student_id=1001735333

To obtain the JSON output use:

http://127.0.0.1:8000/?student_id=1001735333&output=json


## Backend
Use the API documentation at 
* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
Click the "GET" button and click again the "Try it out" button. Then gives the identification on the "`student_id`" box, and fill `json` in the output box. Finally clicj the "Execute" button.

## Frontend
The static frontend reading the last `static/data/filtered.json` file can also be obtained with
```
$ python3 -m http.server 8001
```

See:
* https://www.geeksforgeeks.org/how-to-use-the-javascript-fetch-api-to-get-data/
* https://stackoverflow.com/questions/12460378/how-to-get-json-from-url-in-javascript
* https://stackoverflow.com/a/22790025/2268280
