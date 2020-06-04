# StochQuant

tauhost.c kompilieren und dann über taumain.py oder taumain_windows.py aufrufen!

Zum Kompilieren

```
gcc tauhost.c -o tauhost.o -l OpenCL
```
oder, falls OpenCL nicht global zugreifbar ist

```
gcc tauhost.c -o tauhost.o -I<Pfad zum Include-Verzeichnis> -L<Pfad zum Library-Verzeichnis> -l OpenCL
```
ausführen!

Unter Windows kann im Library-Verzeichnis von reimp oder dlltool (mingw-utils) eine Library erzeugt werden, falls dies nicht funktioniert:

```
reimp OpenCL.lib
```
Falls dies auch nicht funktioniert, die OpenCL.def verwenden und
```
dlltool -l libOpenCL.a -d OpenCL.def -A -k
```
ausführen.
