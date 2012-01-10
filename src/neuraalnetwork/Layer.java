package neuraalnetwork;

import java.util.List;

public class Layer {
	
	private Layer prevLayer;
	private List<Neuron> neurons;
	
	public void forwardPropagation() {
		
	}
	
	public void backPropagation() {
		
	}

	public Layer getPrevLayer() {
		return prevLayer;
	}

	public void setPrevLayer(Layer prevLayer) {
		this.prevLayer = prevLayer;
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public void setNeurons(List<Neuron> neurons) {
		this.neurons = neurons;
	}

}
