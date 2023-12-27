/*!\file CTrieDec.cpp
 * \brief
 *
 *  Created on: 21.06.2019
 *      Author: tombr
 */

#include "CTrieDec.hpp"

CTrieDec::CTrieDec() {
	for(int i = 0; i < LZW_DICT_SIZE; i++) {			//Dictionary initialisieren
		if(i < 256) {
			m_symbolTable[i].setParent(-1);
			m_symbolTable[i].setSymbol(intToString(i));
		}
		else
			m_symbolTable[i].setParent(-2);
	}
}

CTrieDec::~CTrieDec() {

}

string CTrieDec::decode(const vector<unsigned int>& encoded) {
	string decoded("");									  //Rückgabe-String
	string newString("");
	string newSymbol("");
	CDoubleHashing& hash = CDoubleHashing::getInstance(); //Singlelton
	CForwardCounter attemptCounter;						  //Zähler für Hash-Versuche
	bool hashAgain(true);
	unsigned int hashPos(0);							  //Um den Hashwert festzuhalten
	unsigned int parentPos(0);							  //
	vector<unsigned int>::const_iterator it(encoded.begin());

	if(!encoded.empty()) {								//if-guard, welcher verhindert, dass ein leerer Vektor eingelesen wird
		decoded += m_symbolTable[*it].getSymbol();		//ersten Eintrag blind einlesen
		parentPos = *it;
		it++;
		for(; it != encoded.end(); it++) {									//for-Schleife, welche durch den Vektor iteriert
			if(m_symbolTable[*it].getParent() != -2)						//if-guard, welcher vehindert, dass bei Zeichenfolgen, wie "abababababababababa" Probleme auftreten
				newSymbol = *((m_symbolTable[*it].getSymbol()).begin());	//es kann passieren, dass ein Eintrag gelesen wird, welcher noch gar nicht in der m_symbolTable ist
			else
				newSymbol = *((m_symbolTable[parentPos].getSymbol()).begin());

			attemptCounter.setValue(0);  									//attemptCounter auf 0 setzten für die Anzahl der Rehashing-Versuche
			hashAgain = true;												//hashAgain wieder auf wahr setzen, damit er in die while-Schleife geht

			while(hashAgain) {
				hashPos = hash.hash(parentPos, static_cast<int> (static_cast<char>(*newSymbol.begin())), LZW_DICT_SIZE, attemptCounter.getValue());  //neuen Hash-Wert errchenen mit CDoubleHashing

				if(m_symbolTable[hashPos].getParent() == -2) {											//Schaut, ob der Behälter frei ist, also ob es zu einer Kollision gekommen ist oder nicht
					m_symbolTable[hashPos].setSymbol(m_symbolTable[parentPos].getSymbol() + newSymbol);	//neues Symbol
					m_symbolTable[hashPos].setParent(parentPos);										//Elternposition
					parentPos = *it;			//Elternposition neu setzen
					hashAgain = false;			//Es kam zu keiner Kollision, also kein rehashing
				} else {
					hashAgain = true;			//falls es zu einer Kollision kommt muss man rehashen
					attemptCounter.count();		//Zähler um 1 erhöhen
				}
			}
			decoded += m_symbolTable[*it].getSymbol();	//Rückgabe-String erweitern
		}
	}
	return decoded; //String zurückgeben
}
