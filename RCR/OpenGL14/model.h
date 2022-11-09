#ifndef MODEL_H
#define MODEL_H

#include <QString>
#include <QStringList>

#include <QOpenGLFunctions_1_0>
#include <QOpenGLFunctions_4_5_Core>

class DataRow {
public:
    bool normals = false;
    bool textures = false;

    std::array<float, 3> vertex{};
    std::array<float, 2> texture{};
    std::array<float, 3> normal{};

    friend bool operator==(const DataRow &lhs, const DataRow &rhs) {
        return lhs.vertex == rhs.vertex && lhs.texture == rhs.texture && lhs.normal == rhs.normal;
    }

    friend bool operator<(const DataRow &lhs, const DataRow &rhs) {
        return lhs.vertex < rhs.vertex || lhs.texture < rhs.texture || lhs.normal < rhs.normal;
    }
};

class Model : public QOpenGLFunctions_4_5_Core {
public:
    Model() : v_cnt(0), vn_cnt(0), f_cnt(0), read_normals(false), read_textures(false), v(0), vn(0), vt(0) {
        initializeOpenGLFunctions();
    }

    ~Model() = default;


    void readFile(QString fname, bool readNormals, bool readTextures, float scale);

    int getVertDataStride() { return stride; }

    int getVertDataCount() { return f_cnt; }

    int getVertDataSize() { return 3 * f_cnt * stride * sizeof(float); }

    GLuint getVBO() const;

    GLuint getEBO() const;

private:
    QStringList source;
    int v_cnt, vn_cnt, vt_cnt, f_cnt, stride;
    bool read_normals, read_textures;
    float *v, *vn, *vt;

    GLuint VBO;
    GLuint EBO;

    void count_items();

    void alloc_items();

    void parse_items(float scale);

    void print();
};

#endif // MODEL_H
