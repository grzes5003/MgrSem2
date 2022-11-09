#include "model.h"

#include <QFile>
#include <QTextStream>
#include <QDebug>


void Model::readFile(QString fname, bool readNormals, bool readTextures, float scale) {
    qDebug() << "Czytam '" << fname << "'...";

    read_normals = readNormals;
    read_textures = readTextures;

    QFile file(fname);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        throw QString("Nie moge otworzyc pliku '%1'").arg(fname);

    QTextStream ts(&file);
    source.clear();
    while (!ts.atEnd())
        source << ts.readLine();
    file.close();

    count_items();
    alloc_items();
    parse_items(scale);
}

GLuint Model::getVBO() const {
    return VBO;
}

GLuint Model::getEBO() const {
    return EBO;
}

void Model::count_items() {
    v_cnt = vn_cnt = vt_cnt = f_cnt = 0;
    for (int i = 0; i < source.count(); i++) {
        if (source[i].startsWith("v "))
            v_cnt++;
        else if (source[i].startsWith("vn "))
            vn_cnt++;
        else if (source[i].startsWith("vt "))
            vt_cnt++;
        else if (source[i].startsWith("f "))
            f_cnt++;
    }
    qDebug() << "vertices:" << v_cnt;
    qDebug() << "normals:" << vn_cnt;
    qDebug() << "textures:" << vt_cnt;
    qDebug() << "faces:" << f_cnt;
}


void Model::alloc_items() {
    v = new float[3 * v_cnt];
    memset(v, 0, sizeof(float) * 3 * v_cnt);
    if (read_normals) {
        vn = new float[3 * vn_cnt]();
        memset(vn, 0, sizeof(float) * 3 * vn_cnt);
    }
    if (read_textures) {
        vt = new float[2 * vt_cnt]();
        memset(vt, 0, sizeof(float) * 2 * vt_cnt);
    }

    stride = 3 + 3 * int(read_normals) + 2 * int(read_textures);
}


void Model::parse_items(float scale) {
    QString l;
    QStringList sl, sl2;

    // wierzcholki...
    int p = 0;
    for (int i = 0; i < source.count(); i++) {
        if (source[i].startsWith("v ")) {
            l = source[i].mid(2).trimmed();
            sl = l.split(" ");
            v[3 * p + 0] = sl[0].toFloat() * scale;
            v[3 * p + 1] = sl[1].toFloat() * scale;
            v[3 * p + 2] = sl[2].toFloat() * scale;
            p++;
        }
    }

    // normalne...
    if (read_normals) {
        int p = 0;
        for (int i = 0; i < source.count(); i++) {
            if (source[i].startsWith("vn ")) {
                l = source[i].mid(3).trimmed();
                sl = l.split(" ");
                vn[3 * p + 0] = sl[0].toFloat();
                vn[3 * p + 1] = sl[1].toFloat();
                vn[3 * p + 2] = sl[2].toFloat();
                p++;
            }
        }
    }

    // wspolrzedne tekstur...
    if (read_textures) {
        int p = 0;
        for (int i = 0; i < source.count(); i++) {
            if (source[i].startsWith("vt ")) {
                l = source[i].mid(3).trimmed();
                sl = l.split(" ");
                vt[2 * p + 0] = sl[0].toFloat();
                vt[2 * p + 1] = sl[1].toFloat();
                p++;
            }
        }
    }

    // trojkaty...
    p = 0;

    std::vector <DataRow> data_rows;
    std::map <DataRow, GLuint> vert_data_map;
    std::vector <GLuint> triangle_indices;

    for (int i = 0; i < source.count(); i++) {
        if (source[i].startsWith("f ")) {
            l = source[i].mid(2).trimmed();
            sl = l.split(" ");

            for (int j = 0; j < 3; j++) {
                sl2 = sl[j].split("/");
                while (sl2.count() < 3)
                    sl2.append("");

                int vi = sl2[0].toInt() - 1;

                DataRow data_row{};

                data_row.vertex[0] = v[3 * vi + 0];
                data_row.vertex[1] = v[3 * vi + 1];
                data_row.vertex[2] = v[3 * vi + 2];

                if (read_normals) {
                    data_row.normals = true;
                    int vni = sl2[2].toInt() - 1;
                    data_row.normal[0] = vn[3 * vni + 0];
                    data_row.normal[1] = vn[3 * vni + 1];
                    data_row.normal[2] = vn[3 * vni + 2];
                }

                if (read_textures) {
                    data_row.textures = true;
                    int vti = sl2[1].toInt() - 1;
                    data_row.texture[0] = vt[2 * vti + 0];
                    data_row.texture[1] = vt[2 * vti + 1];
                }

                if (vert_data_map.count(data_row) == 0) {
                    vert_data_map[data_row] = p;
                    data_rows.push_back(data_row);
                    triangle_indices.push_back(p);
                    p += 1;
                } else {
                    triangle_indices.push_back(vert_data_map.find(data_row)->second);
                }

            }
        }
    }
    size_t i = 0;

    const size_t transfer_size = stride * data_rows.size();
    float *transfer_data = new float[transfer_size];
    for (const auto &row: data_rows) {
        transfer_data[i] = row.vertex[0];
        transfer_data[i + 1] = row.vertex[1];
        transfer_data[i + 2] = row.vertex[2];

        if (row.normals) {
            transfer_data[i + 3] = row.normal[0];
            transfer_data[i + 4] = row.normal[1];
            transfer_data[i + 5] = row.normal[2];
        }

        if (row.textures) {
            transfer_data[i + 6] = row.texture[0];
            transfer_data[i + 7] = row.texture[1];
        }

        i += stride;
    }

    delete[] v;
    delete[] vn;
    delete[] vt;

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, transfer_size * sizeof(float), transfer_data, GL_STATIC_DRAW);

    delete[] transfer_data;

    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangle_indices.size() * 3 * sizeof(GLuint), triangle_indices.data(),
                 GL_STATIC_DRAW);

    qDebug() << "Ok, model wczytany.";
}

void Model::print() {
    qDebug() << "stride:" << stride;
}

