/*!\file CArrayEnc.cpp
 * \brief
 *
 *  Created on: 13.06.2019
 *      Author: tombr
 */

#include "CArrayEnc.hpp"

CArrayEnc::CArrayEnc(): currentSize(256) { 					//initialiesierung mit der Zahl 256 welches der Groesse des Arrays entspricht

	for(int i = 0; i < 256; i++) { 							//geht durch das ganze Array
		m_symbolTable[i].setSymbol(intToString(i)); 		//Zuweisung der ersten 256 Symbole der ACII Tabelle
	}
	//cout << "CArrayEnc funzt" << endl;
}

CArrayEnc::~CArrayEnc() {

}

int CArrayEnc::searchInTable(const string& compare) {	 	//sucht nach in der Tabelle nach dem passenden string

	for(int i = 0; i < currentSize; i++) { 					// Schleife die durch das Array geht
		if(m_symbolTable[i].getSymbol() == compare) 		//Vergleichen von Elementen
			return i; 										//Position als Rückgabewert
	}
	return -1; 												//wenn string nicht gefunden return value = -1
}

void CArrayEnc::printSymbolTable(unsigned int index) {
	for(unsigned int i = index; i < static_cast<unsigned int> (currentSize); i++) {
		cout << m_symbolTable[i].getSymbol() << " ";
	}
	cout << endl;
}

vector<unsigned int> CArrayEnc::encode(const string& a) {
	string newString("");
	string saveIt;
	int previousVal(-2); 									//speichert die position von der alten Zeichenkette
	int searched(-2); 										//Rückgabe variable für searchInTable()
	int vec(0); 											//Wo bin ich bei meinem Rückgabevector?
	vector<unsigned int> encoded; 							//Rückgabe vector

	if(a != "") {

	for(string::const_iterator it = a.begin(); it != a.end();) { 	//For-Schleife, welche durch den gegeben string a iteriert, bis zum ende
		newString += *it;											//Fügt das nächste Zeichen an von a an newString an.
		//cout << newString << " ";
		searched = searchInTable(newString); 						//Schaut ob es newString gibt
		//cout << searched << endl;

		if(searched < 0) {
			encoded.push_back(static_cast<unsigned int> (previousVal));
			//cout << "Ich habs bis hier geschafft" << endl;
			m_symbolTable[currentSize].setSymbol(newString);
			currentSize++;
			vec++;
			newString = "";
		}
		else {
			previousVal = searched;
			it++;
		}
	}
	encoded.push_back(static_cast<unsigned int> (previousVal));
	//cout << "Encoder funzt" << endl;
	}
	return encoded; 										//Gibt den Vector zurück
}
