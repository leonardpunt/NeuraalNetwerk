package images;

import java.io.File;
import java.io.IOException;

import mnist.tools.MnistManager;

public class ImageReader {

	public void test() {
		MnistManager m;
		try {
			m = new MnistManager("data/t10k-images-idx3-ubyte",
					"data/t10k-labels-idx1-ubyte");
			m.setCurrent(10); // index of the image that we are interested in
			int[][] image = m.readImage();
			
			System.out.println("Label:" + m.readLabel());
			new File("output").mkdir();
			MnistManager.writeImageToPpm(image, "output/10.ppm");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
