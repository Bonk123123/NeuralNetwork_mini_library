// ThroughTheValley.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include "crow.h"
#include "crow/middlewares/cors.h"
#include "AI.h"
#include <iostream>

using namespace std;

// функция для чтения данных
vector<vector<float>> read_mnist(string full_path, int quantityNumbers)
{
    ifstream file(full_path);
    string label;
    vector<vector<float>> mnist;
    if (file.is_open())
    {
        file >> label;
        string stroke;
        int j = 0;
        while (file >> stroke && j != quantityNumbers)
        {
            vector<float> number;
            string num = "";
            file >> stroke;
            for (size_t i = 0; i < stroke.size(); i++)
            {
                if (i == 1)
                {
                    number.push_back((atof(num.c_str())));
                }
                if (stroke[i] != ',' && stroke[i] != ' ')
                {
                    num.push_back(stroke[i]);
                }
                else
                {
                    number.push_back(atof(num.c_str()));
                    num = "";
                }


            }
            mnist.push_back(number);
            number.clear();
            j++;
        }
    }

    return mnist;
}
//

int main()
{
    // создание архитектуры нейросети
    AI neo(785, 1);

    neo.addLayer(55, 0, "sigmoid");
    neo.addLayer(30, 0, "sigmoid");
    neo.addLayer(10, 0, "sigmoid");
    //

    // чтение данных
    vector<vector<float>> mnistData(read_mnist("C:/Users/Arthur/source/repos/ThroughTheValley/ThroughTheValley/mnist/mnist_train.csv", 60000));
    vector<vector<float>> mnistTest(read_mnist("C:/Users/Arthur/source/repos/ThroughTheValley/ThroughTheValley/mnist/mnist_test.csv", 1000));

    vector<vector<float>> mnistAnswer;
    vector<vector<float>> mnistTestAnswer;
    //

    // заполнение правильных ответов для данных из общих данных
    for (size_t i = 0; i < mnistTest.size(); i++)
    {
        mnistTestAnswer.push_back(vector<float> { mnistTest[i][0] });
        mnistTest[i].erase(mnistTest[i].begin());
    }
    //


    // представление правильных ответов в виде в 1 и 0
    for (size_t i = 0; i < mnistData.size(); i++)
    {
        vector<float> number;
        //number.push_back(mnistData[i][0]);

        for (size_t j = 0; j < 10; j++)
        {

            if (mnistData[i][0] == j)
            {
                number.push_back(1);
            }
            else
            {
                number.push_back(0);
            }
        }



        mnistData[i].erase(mnistData[i].begin());
        mnistAnswer.push_back(number);
        number.clear();
        //cout << 1 / (mnistData[i][0] + 1) << ' ';
    }
    //


    // нахождение среднего и максимального в тренировочных данных
    float max = 0;
    float avg = 0;
    for (size_t i = 0; i < mnistData.size(); i++)
    {
        for (size_t j = 0; j < mnistData[i].size(); j++)
        {
            avg += mnistData[i][j];
            if (mnistData[i][j] > max)
            {
                max = mnistData[i][j];
            }
        }
    }
    avg /= mnistData.size() * mnistData[0].size();
    //

    // нормализация тренировочных данных
    for (size_t i = 0; i < mnistData.size(); i++)
    {
        for (size_t j = 0; j < mnistData[i].size(); j++)
        {
            mnistData[i][j] = (mnistData[i][j] - avg) / max;
            //mnistData[i][j] = (mnistData[i][j]) / max;
        };
    }
    //

    // нормализация тестовых данных
    for (size_t i = 0; i < mnistTest.size(); i++)
    {
        for (size_t j = 0; j < mnistTest[i].size(); j++)
        {
            mnistTest[i][j] = (mnistTest[i][j] - avg) / max;
            //mnistTest[i][j] = (mnistTest[i][j]) / maxtest;
        }
    }

    // отображение тестов в консоль
    /*
    for (size_t i = 0; i < mnistTestAnswer.size(); i++)
    {
        vector<float> answer(neo.predict(mnistTest[i]));
        for (size_t j = 0; j < answer.size(); j++)
        {
            cout << answer[j] << "\t" << '|' << '\t' << mnistTestAnswer[i][0] << endl;
        }
        cout << endl;
    }
    */
    //

    // выбор оптимайзера и размера батча ( sgd, sgdnest, adagrad, rms, adadelta, adam )
    neo.setOptimizer("adam");
    neo.setBatchSize(3); // по умолчанию 1
    //

    // тренировка ( данные, желаемый выход, количество эпох )
    neo.train(mnistData, mnistAnswer, 3000);
    //


    // отображение тестов в консоль
    for (size_t i = 0; i < mnistTestAnswer.size(); i++)
    {
        vector<float> answer(neo.predict(mnistTest[i]));
        for (size_t j = 0; j < answer.size(); j++)
        {
            cout << answer[j] << "\t" << '|' << '\t' << mnistTestAnswer[i][0] << endl;
        }
        cout << endl;
    }
    //
    

    // Enable CORS
    crow::App<crow::CORSHandler> app;

    // Customize CORS
    auto& cors = app.get_middleware<crow::CORSHandler>();

    // clang-format off
    
    cors
      .global()
        .headers("X-Custom-Header", "Upgrade-Insecure-Requests")
        .methods("POST"_method, "GET"_method)
      .prefix("/cors")
        .origin("example.com")
      .prefix("/nocors")
        .ignore();
    //

    //crow::SimpleApp app; //define your crow application
    
    //define your endpoint at the root directory
    CROW_ROUTE(app, "/define-number").methods(crow::HTTPMethod::POST)([&](const crow::request& req)
        {
            try
            {
                vector<float> data;
                auto number = crow::json::load(req.body);
                //cout << number["colors"] << endl;
                for (size_t i = 0; i < number["colors"].size(); i++)
                {
                    data.push_back(((number["colors"][i]).d() - avg) / max);
                }

                vector<float> answer(neo.predict(data));

                int otvet = 0;
                float probability = 0;
                for (size_t i = 0; i < answer.size(); i++)
                {
                    if (answer[i] > probability)
                    {
                        otvet = i;
                        probability = answer[i];
                    }
                }

                return crow::response(200, to_string(otvet));
            }
            catch (const std::exception& ex)
            {
                return crow::response(400, "error");
            }
            
        });

    //set the port, set the app to run on multiple threads, and run the app
    app.port(18080).multithreaded().run();
    

}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
