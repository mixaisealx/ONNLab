#pragma once

#include <atomic>
#include <thread>

class AtomicSpinlock {
	std::atomic_flag locker;

public:
	inline void lock() {
		while (locker.test_and_set(std::memory_order_acquire)) {
			std::this_thread::yield();
		}
	}

	inline void unlock() {
		locker.clear(std::memory_order_release);
	}
};