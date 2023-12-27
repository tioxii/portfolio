/*!\file CCounter.cpp
 * \brief
 *
 *  Created on: 15.06.2019
 *      Author: tombr
 */

#include "CCounter.hpp"

CCounter::CCounter(): m_value(0) {

}

CCounter::~CCounter() {

}

int CCounter::getValue() const{
	return m_value;
}

void CCounter::setValue(int value) {
	m_value = value;
}
