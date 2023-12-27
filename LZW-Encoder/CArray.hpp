/*!\file CArray.hpp
 * \brief
 *
 *  Created on: 13.06.2019
 *      Author: tombr
 */

#ifndef CARRAY_HPP_
#define CARRAY_HPP_

#include "XOutOfBounds.hpp"

template <typename T, unsigned int N> //Template

class CArray {
public:
	//Konstruktor
	CArray<T,N>() {
		entries = new T[N];
	}

	//Kopierkonstruktor
	CArray(CArray<T, N>& other) {
		entries = new T[N];

		for(unsigned int i = 0; i < N; i++) {
			entries[i] = other[i];
		}
	}

	//Destruktor
	~CArray() {
		delete[] entries;
		entries = nullptr;
	}

	//operator=
	void operator=(CArray& other) {
		for(unsigned int i = 0; i < N; i++) {
			entries[i] = other [i];
		}
	}

	//operator[]
	T& operator[](unsigned int index) {
		if(index >= N)
			throw(XOutOfBounds("Ueberschreitung des Index"));
		else
			return entries[index];
	}

private:
	T* entries; //Array
};

#endif /* CARRAY_HPP_ */
