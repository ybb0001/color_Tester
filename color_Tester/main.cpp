#include "color_Tester.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	color_Tester w;
	w.show();
	return a.exec();
}
