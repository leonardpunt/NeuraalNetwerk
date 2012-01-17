package images;

import java.io.IOException;

import mnist.tools.MnistManager;

public class ImageReader {

	MnistManager trainingSet;
	MnistManager testSet;

	public ImageReader() {
		try {
			trainingSet = new MnistManager("data/train-images-idx3-ubyte",
					"data/train-labels-idx1-ubyte");
			testSet = new MnistManager("data/t10k-images-idx3-ubyte",
					"data/t10k-labels-idx1-ubyte");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public double[] readImage(int i, MnistManager set) {
		set.setCurrent(i); // index of the image that we are interested in
		int[][] image = new int[28][28];
		double[] newImage = new double[784];

		try {
			image = set.readImage();
			int count = 0;
			for (int j = 0; j < image.length; j++) {
				for (int k = 0; k < image[j].length; k++) {
					newImage[count] = ((double) image[j][k] / 128.0) - 1.0;
					count++;
				}
			}			
			return newImage;
		} catch (IOException e) {
			e.printStackTrace();
		}

		return null;
	}
	
	public int readLabel(int i, MnistManager set) {
		set.setCurrent(i);		
		try {
			return set.readLabel();
		} catch (IOException e) {
			e.printStackTrace();
		}		
		return 0;
	}

	public int lengthOfTrainingSet() {
		return trainingSet.getImages().getCount();
	}

	public int lengthOfTestSet() {
		return testSet.getImages().getCount();
	}
	
	public MnistManager getTrainingSet() {
		return trainingSet;
	}

	public MnistManager getTestSet() {
		return testSet;
	}
}
