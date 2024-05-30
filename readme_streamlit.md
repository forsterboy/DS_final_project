
- create a useful web app combining machine learning and data science, 
    that can display data and collect needed parameters for predicting car price based on applied models 



Car price prediction web app take the preprocessed dateset , create for varios feateres dividing a continuous variable into a set of intervals.

create the models comparison

Additionally we created for varios feateres Binning integrate continuous variable to simplify by reducing the number of unique values.


Our mission is to prepare a web app to make it available in production.



sorry, i have to go with my child to hospital 

ValueError: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead.
Traceback:
File "/Users/nikoma/Documents/Greenbootcamps/DS_final_project/.venv/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 600, in _run_script
    exec(code, module.__dict__)
File "/Users/nikoma/Documents/Greenbootcamps/DS_final_project/car_price_streamlet_app.py", line 447, in <module>
    main()
File "/Users/nikoma/Documents/Greenbootcamps/DS_final_project/car_price_streamlet_app.py", line 406, in main
    refreshModels(tab)
File "/Users/nikoma/Documents/Greenbootcamps/DS_final_project/car_price_streamlet_app.py", line 313, in refreshModels
    lineGraph_prediction(50, Y_train, testPredictions[key])
File "/Users/nikoma/Documents/Greenbootcamps/DS_final_project/car_price_streamlet_app.py", line 274, in lineGraph_prediction
    plt.plot(aa, y[:number], marker='.', label="actual")









