/*!\file CDoubleHashing.cpp
 * \brief
 *
 *  Created on: 18.06.2019
 *      Author: tombr
 */

#include "CDoubleHashing.hpp"

CDoubleHashing CDoubleHashing::m_instance;

CDoubleHashing::CDoubleHashing() {

}

CDoubleHashing::~CDoubleHashing() {

}

CDoubleHashing& CDoubleHashing::getInstance() {
	return m_instance;							//Einzige CDoubleHashing-Instanz zurückgeben
}

unsigned int CDoubleHashing::hash(unsigned int I, unsigned int J, unsigned int dict_size, unsigned int attempt) {
	unsigned int x((I + J)*(I + J + 1)); //Cantorische Paarungsfunktion
	unsigned int ret(0); 				 //return-value

	x /= 2; //Erweiterung der Paarungsfunktion (ganze)
	x += J;

	ret = (x + (attempt*(1 + (x % (dict_size - 2))))) % dict_size; //Hash-Funktion

	return ret;
}


