#pragma once
#include <vector>

namespace nn::interfaces
{
	class ConnectionBasicInterface;

	class NeuronBasicInterface {
		friend class ConnectionBasicInterface;

	protected:
		virtual void AddInputConnection(ConnectionBasicInterface *) = 0;
		virtual void AddOutputConnection(ConnectionBasicInterface *) = 0;
		virtual void RemoveInputConnection(ConnectionBasicInterface *) = 0;
		virtual void RemoveOutputConnection(ConnectionBasicInterface *) = 0;

	public:
		NeuronBasicInterface() = default;
		virtual ~NeuronBasicInterface() = default;

		virtual const std::vector<ConnectionBasicInterface *> &InputConnections() = 0;
		virtual const std::vector<ConnectionBasicInterface *> &OutputConnections() = 0;

		virtual void UpdateOwnLevel() = 0;
		virtual float OwnLevel() = 0;

		virtual bool IsTrainable() = 0;

		virtual float ActivationFunction(float x) const = 0;
		virtual float ActivationFunctionDerivative(float x) const = 0;
	};
	
	class ConnectionBasicInterface {
	protected:
		inline void NBI_AddInputConnection(NeuronBasicInterface *nbi, ConnectionBasicInterface *cbi) {
			nbi->AddInputConnection(cbi);
		}

		inline void NBI_AddOutputConnection(NeuronBasicInterface *nbi, ConnectionBasicInterface *cbi) {
			nbi->AddOutputConnection(cbi);
		}

		inline void NBI_RemoveInputConnection(NeuronBasicInterface *nbi, ConnectionBasicInterface *cbi) {
			nbi->RemoveInputConnection(cbi);
		}

		inline void NBI_RemoveOutputConnection(NeuronBasicInterface *nbi, ConnectionBasicInterface *cbi) {
			nbi->RemoveOutputConnection(cbi);
		}

	public:
		ConnectionBasicInterface() = default;
		virtual ~ConnectionBasicInterface() = default;

		virtual NeuronBasicInterface *From() = 0;
		virtual NeuronBasicInterface *To() = 0;

		virtual float Weight() = 0;
		virtual void Weight(float weight) = 0;
	};

	using NBI = NeuronBasicInterface;
	using CBI = ConnectionBasicInterface;

	template<typename T>
	concept NeuronInherit = std::is_base_of<nn::interfaces::NeuronBasicInterface, T>::value;

	template<typename T>
	concept ConnectionInherit = std::is_base_of<nn::interfaces::ConnectionBasicInterface, T>::value;
}
