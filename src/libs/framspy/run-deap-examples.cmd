rem To learn about all available options of the .py algorithm below, add "-h" to its parameters.
rem Use the source code of the examples as a starting point for your customizations.
rem Example usage:

set DIR_WITH_FRAMS_LIBRARY=../Framsticks50rc29/


rem simple one-criterion evolution, maximize the number of neurons
start ../venv/Scripts/python.exe FramsticksEvolution.py   -path %DIR_WITH_FRAMS_LIBRARY%   -sim "eval-allcriteria.sim"  -opt numneurons
 

rem as above but "chaining" .sim files, subsequent files overwrite selected parameters
start ../venv/Scripts/python.exe FramsticksEvolution.py   -path %DIR_WITH_FRAMS_LIBRARY%   -sim "eval-allcriteria.sim;deterministic.sim;sample-period-longest.sim"    -opt velocity


rem introducing hard limit on the number of Parts, using f9 genetic encoding and saving Hall of Fame
start ../venv/Scripts/python.exe FramsticksEvolution.py   -path %DIR_WITH_FRAMS_LIBRARY%   -opt velocity   -max_numparts 6    -genformat 9   -hof_savefile HoF-f9.gen


rem two criteria
start ../venv/Scripts/python.exe FramsticksEvolution.py   -path %DIR_WITH_FRAMS_LIBRARY%   -popsize 40    -generations 10    -opt velocity,vertpos
