
# Chamfer distance

- src: C source code
- functions: the autograd functions
- modules: code of the nn module
- build.py: a small file that compiles your module to be ready to use
- test.py: an example file that loads and uses the extension

```bash
cd src
nvcc -c -o nnd_cuda.cu.o nnd_cuda.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ..
python build.py
python test.py
```
