/*!\file CKnot.cpp
 * \brief
 *
 *  Created on: 15.06.2019
 *      Author: tombr
 */

#include "CKnot.hpp"

CKnot::CKnot(): CEntry(), m_parent(-2) { //Initialisierungsliste

}

CKnot::~CKnot() {

}

int CKnot::getParent() const {
	return m_parent;					//m_parent zurückgeben
}

void CKnot::setParent(int parent) {
	m_parent = parent;					//m_parent setzen
}
