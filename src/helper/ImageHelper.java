package helper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import mnist.tools.MnistManager;

public class ImageHelper {

	MnistManager trainingSet;
	MnistManager testSet;

	public ImageHelper() {
		try {
			trainingSet = new MnistManager("data/train-images-idx3-ubyte",
					"data/train-labels-idx1-ubyte");
			testSet = new MnistManager("data/t10k-images-idx3-ubyte",
					"data/t10k-labels-idx1-ubyte");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public int[] readImage(int i, MnistManager set) {
		set.setCurrent(i); // index of the image that we are interested in
		int[][] image = new int[28][28];
		int[] newImage = new int[784];

		try {
			image = set.readImage();
			for (int j = 0; j < image.length; j++) {
				for (int k = 0; k < image[j].length; k++) {
					newImage[k + j * image.length] = image[j][k];
				}
			}
			OtsuTresholdingAlgorithm ota = new OtsuTresholdingAlgorithm();			
			return ota.doThreshold(newImage);
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
	
	public List<Integer> getIndicesValidationSet(int sizeValidationSet) {
		ArrayList<Integer> indicesValidationSet = new ArrayList<Integer>();
		for (int i = 1; i <= sizeValidationSet; i++) {
			indicesValidationSet.add((int) (Math.random() * lengthOfTrainingSet()));
		}
		return indicesValidationSet;
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
