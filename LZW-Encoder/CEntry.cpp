/*!\file CEntry.cpp
 * \brief
 *
 *  Created on: 14.06.2019
 *      Author: tombr
 */

#include "CEntry.hpp"

unsigned int CEntry::m_number = 0;

CEntry::CEntry(): m_symbol("") {
	m_number++;
}

CEntry::~CEntry() {
	m_number--;
}

const string& CEntry::getSymbol()const {
	return m_symbol;
}

void CEntry::setSymbol(string symbol) {
	m_symbol = symbol;
}

unsigned int CEntry::getNumber() {
	return m_number;
}
