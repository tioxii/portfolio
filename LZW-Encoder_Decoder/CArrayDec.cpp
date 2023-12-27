/*!\file CArrayDec.cpp
 * \brief
 *
 *  Created on: 17.06.2019
 *      Author: tombr
 */

#include "CArrayDec.hpp"

CArrayDec::CArrayDec(): currentSize(256) {

	for(int i = 0; i < LZW_DICT_SIZE; i++) { 				//geht durch das ganze Array
			if(i < 256)
				m_symbolTable[i].setSymbol(intToString(i)); //Zuweisung der ersten 256 Symbole der ACII Tabelle
			else
				m_symbolTable[i].setSymbol(" ");
	}
}

CArrayDec::~CArrayDec() {

}

int CArrayDec::searchInTable(const string& compare) { //sucht nach in der Tabelle nach dem passenden string

	for(int i = 0; i < currentSize; i++) { 			// Schleife die durch das Array geht
		if(m_symbolTable[i].getSymbol() == compare) //Vergleichen von Elementen
			return i; 								//Position als Rückgabewert
	}
	return -1; //wenn string nicht gefunden return value = -1
}

string CArrayDec::decode(const vector<unsigned int>& encoded) {
	string decoded("");
	string oldString("");
	string newString("");
	vector<unsigned int>::const_iterator it = encoded.begin();

	if(!encoded.empty()) { 										 //if-guard verhindert, dass ein leerer Vektor eingelsen wird
		oldString = m_symbolTable[*it].getSymbol();				 //erstets Zeichen blind einlesen
		decoded += m_symbolTable[*it].getSymbol();				 //und zum ausgabe-vektor hinzufügen
		it++;

		for(; it != encoded.end(); it++) {
			if((*it) == static_cast<unsigned int> (currentSize)) //if-Anweisung dient für den Sonderfall, falls die Zeichenkette z.B "abababababababababababa" lautet
				newString = *(oldString.begin());				 //rekonstruierung mithilfe der alten Zeichenkette
			else
				newString = m_symbolTable[*it].getSymbol();
			m_symbolTable[currentSize].setSymbol(oldString + *(newString.begin())); //ergänzung der Tabelle um weitere Zeichen
			currentSize++;														    //um die Größe der aktuellen Tabelle zu speichern
			decoded += m_symbolTable[*it].getSymbol();								//Ausgabe String
			oldString = m_symbolTable[*it].getSymbol();
		}
	}
	return decoded;	 //decodierter String als Rückgabewert
}
