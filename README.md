# python_roctx

Example: Using roctx calls within a Python program


Step 1: Compile the hip_code library
```
module load rocm/6.0.0
make
```

Step 2: Run the script to make sure it works
```
python3 roctx_example.py
```

Step 3: Get the roctx trace using rocprof
```
rocprof --roctx-trace -d rocprof_output -o rocprof_output/results.csv python3 roctx_example.py
```