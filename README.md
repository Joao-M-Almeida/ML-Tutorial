# ML-Tutorial

Repository with the presentation "Introduction to Machine Learning with Python and Scikit Learn" to be given at CERN Spring Campus 2017 in Glasgow.


Install python3 and virtualenv, then:

    python3 -m venv MLTutorial
    source bin/activate
    pip install ipykernel
    pip install nose
    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install scikit-learn
    nosetests -v sklearn
    pip install pandas

References:
- http://docs.python-guide.org/en/latest/dev/virtualenvs/
- http://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs


Main dependencies:
- Python 3
- NumPy
- SciPy
- scikit-learn
- matplotlib
- pandas
- nose (for testing scikit-learn)


Whisky Datasets:
- [David Whishart: Whisky Classified dataset](https://www.mathstat.strath.ac.uk/outreach/nessie/nessie_whisky.html)
- [Whisky Analysis](http://whiskyanalysis.com/index.php/2017/01/13/whiskyanalysis-exceeds-1000-whiskies-january-2017/) Meta-Critic [Dataset](http://whiskyanalysis.com/index.php/database/)
- [Whisky Monitor](http://www.whisky-monitor.com/search.jsp) from [Malt Maniacs](http://www.maltmaniacs.net/)
