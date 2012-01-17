package images;

import java.io.File;
import java.io.IOException;

import mnist.tools.MnistManager;

public class ImageReader {

    MnistManager m;

    public ImageReader() {
        try {
            this.m = new MnistManager("data/t10k-images-idx3-ubyte",
                    "data/t10k-labels-idx1-ubyte");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

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

    public int[][] readImage(int i) {
        m.setCurrent(i); // index of the image that we are interested in
        int[][] image = new int[28][28];
        int[][] newImage = new int[28][28];

        try {
            image = m.readImage();
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int j = 0; j < image.length; j++) {
            for (int k = 0; k < image[j].length; k++) {
                newImage[j][k] = (image[j][k]/128)-1;
            }
        }

        return newImage;
    }
}
