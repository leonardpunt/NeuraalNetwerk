package neuraalnetwork;

public class Connection {
	
	private Neuron connectionTo;
	private double weight;

	Connection(Neuron connectionTo, double weight) {
		this.connectionTo = connectionTo;
		this.weight = weight;
	}
	
	public Neuron getConnectionTo() {
		return connectionTo;
	}

	public double getWeight() {
		return weight;
	}
}
