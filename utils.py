
import matplotlib.pyplot as plt

def plot_history(history, hist1, hist2, title, x_label, y_label):

	plt.plot(history.history[hist1])
	plt.plot(history.history[hist2])
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend(['train','test'], loc='upper left')
	plt.show()