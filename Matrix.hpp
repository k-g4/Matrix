#pragma once
#include <math.h>
#include <iostream>
#include <initializer_list>
#include <array>
#include <random>

//定数で初期化するためのテンプレート再帰
template<typename Type, size_t Dim>
struct Init_vec{
    static void init(Type *vector, Type value){
        vector[Dim - 1] = value;
        Init_vec<Type, Dim - 1>::init(vector, value);
    }
};
template<typename Type>
struct Init_vec<Type, 0>{
    static void init(Type *vector, Type value){}
};

//乱数で初期化するためのテンプレート再帰
std::random_device rd;
std::mt19937 gen(rd());
template<typename Type, size_t Dim>
struct Init_vecRan {
    static void init(Type *vector, Type min, Type max) {
        if constexpr (std::is_integral<Type>::value) {
            std::uniform_int_distribution<Type> dis(min, max);
            vector[Dim - 1] = dis(gen);
            Init_vecRan<Type, Dim - 1>::init(vector, min, max);
        } else {
            std::uniform_real_distribution<Type> dis(min, max);
            vector[Dim - 1] = dis(gen);
            Init_vecRan<Type, Dim - 1>::init(vector, min, max);
        }
    }
};
template<typename Type>
struct Init_vecRan<Type, 0> {
    static void init(Type *vector, Type min, Type max) {}
};

//行列
template<typename Type, int Row, int Col>
class Matrix{
private:
    //ベクトル
    template<typename T, size_t Dim>
    class Vector{
    private:
        T *vector;

    public:
        Vector(){
            vector = new T[Dim];
        }

        Vector(T value){
            vector = new T[Dim];
            Init_vec<T, Dim>::init(vector, value);
        }

        Vector(T min, T max){
            vector = new T[Dim];
            Init_vecRan<T, Dim>::init(vector, min, max);
        }

        Vector(std::initializer_list<T> initList){
            if (initList.size() != Dim) {
                throw std::runtime_error("Initializer list size does not match vector size.");
            }
            vector = new T[Dim];
            std::copy(initList.begin(), initList.end(), vector);
        }

        ~Vector(){
            delete[] vector;
        }

        T& operator[](size_t i){
            if(i >= Dim) throw std::out_of_range("Index out of range.");
            return vector[i];
        }

        const T& operator[](size_t i)const{
            if(i >= Dim) throw std::out_of_range("Index out of range.");
            return vector[i];
        }

        Vector& operator=(const Vector& A){
            for(size_t i = 0; i < Dim; i++)
                this->vector[i] = A.vector[i];
            return *this;
        }

        void print() const {
            for (size_t i = 0; i < Dim; ++i) {
                std::cout << vector[i] << " ";
            }
            std::cout << std::endl;
        }
    };
    //行列の確保
    Vector<Vector<Type, Col>, Row> matrix;

public:
    Matrix(Type value){
        for(int i = 0; i < Row; i++){
            matrix[i] = Vector<Type, Col>(value);
        }
    }

    Matrix(Type min, Type max){
        for(int i = 0; i < Row; i++){
            matrix[i] = Vector<Type, Col>(min, max);
        }
    }

    Matrix(std::initializer_list<std::initializer_list<Type>> initList) {
        if (initList.size() != Row) {
            throw std::runtime_error("Initializer list row size does not match matrix row size.");
        }
        int i = 0;
        for (auto rowList : initList) {
            if (rowList.size() != Col) {
                throw std::runtime_error("Initializer list column size does not match matrix column size.");
            }
            matrix[i++] = Vector<Type, Col>(rowList);
        }
    }

    ~Matrix(){}

    Vector<Type, Col>& operator[](size_t i){
        return matrix[i];
    }

    const Vector<Type, Col>& operator[](size_t i)const{
        return matrix[i];
    }

    Type& operator()(size_t i, size_t j){
        return matrix[i][j];
    }

    const Type& operator()(size_t i, size_t j)const{
        return matrix[i][j];
    }

    Matrix& operator=(const Matrix& A){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) = A(i, j);
            }
        }
        return *this;
    }

    template<typename Expr>
    Matrix& operator=(const Expr& e){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) = e(i, j);
            }
        }
        return *this;
    }

    Matrix& operator=(const Type& k){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) = k;
            }
        }
        return *this;
    }

    Matrix& operator+(){
        return *this;
    }

    Matrix& operator-(){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) = -(*this)(i, j);
            }
        }
        return *this;
    }

    Matrix& operator+=(const Matrix& A){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) += A(i, j);
            }
        }
        return *this;
    }

    Matrix& operator+=(const Type k){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) += k;
            }
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& A){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) -= A(i, j);
            }
        }
        return *this;
    }

    Matrix& operator-=(const Type k){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) -= k;
            }
        }
        return *this;
    }

    Matrix& operator*=(const Matrix& A){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) *= A(i, j);
            }
        }
        return *this;
    }

    Matrix& operator*=(const Type k){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) *= k;
            }
        }
        return *this;
    }

    Matrix& operator/=(const Matrix& A){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) /= A(i, j);
            }
        }
        return *this;
    }

    Matrix& operator/=(const Type k){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) /= k;
            }
        }
        return *this;
    }

    Matrix& operator%=(const Matrix& A){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) %= A(i, j);
            }
        }
        return *this;
    }

    Matrix& operator%=(const Type k){
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                (*this)(i, j) %= k;
            }
        }
        return *this;
    }

    bool operator==(const Matrix& A){
        bool result = true;
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                result &= ((*this)(i, j) == A(i, j));
            }
        }
        return result;
    }

    bool operator!=(const Matrix& A){
        return !((*this) == A);
    }

    //転置
    Matrix& T(){
        Matrix<Type, Col, Row> A(0);
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                A(j, i) = (*this)(i, j);
            }
        }
        return A;
    }

    //行列式
    Type det(){
        if(Row != Col) return false;

        Type result = 1;

        if(Row == 2){
            result = (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        }else if(Row == 3){
            result = (*this)(0, 0) * (*this)(1, 1) * (*this)(2, 2)
                   + (*this)(1, 0) * (*this)(2, 1) * (*this)(0, 2)
                   + (*this)(2, 0) * (*this)(0, 1) * (*this)(1, 2)
                   - (*this)(2, 0) * (*this)(1, 1) * (*this)(0, 2)
                   - (*this)(1, 0) * (*this)(0, 1) * (*this)(2, 2)
                   - (*this)(0, 0) * (*this)(2, 1) * (*this)(1, 2);
        }else{
            Matrix<double, Row, Col> A(0);
            A = *this;
            double buf = 0.0;
            Type samp = 0.0;
            for(int i = 0; i < Row; i++){
                //ピボットの選択
                int pivot = i;
                for(int j = i + 1; j < Row; j++){
                    if(abs(A(j, i)) > abs(A(pivot, i)))
                        pivot = j;
                }

                //ピボットが0なら行列式は0
                if(A(pivot, i) == 0)
                    return 0;

                //行の入れ替え
                if(i != pivot){
                    for(int j = 0; j < Row; j++){
                        samp = A(i, j);
                        A(i, j) = A(pivot, j);
                        A(pivot, j) = samp;
                    }
                    result *= -1;
                }

                //上三角化
                for(int j = i + 1; j < Row; j++){
                    buf = A(j, i) / A(i, i);
                    for(int k = i; k < Row; k++)
                        A(j, k) -= A(i, k) * buf;
                }

                result *= A(i, i);
            }
        }
        return result;
    }

    //逆行列
    Matrix<double, Row, Col> inv(){
        if(Row != Col) throw std::runtime_error("matrix size does not match.");

        //行列式が0なら逆行列は定義されない
        if((*this).det() == 0)
            throw std::runtime_error("inverse matrix is not defined.");

        Matrix<double, Row, Col> result(0);
        Matrix<double, Row, Col * 2> expansion(0);
        double samp = 0.0;
        double buf = 0.0;
        //左にもとの行列を代入
        for(int i = 0; i < Row; i++){
            for(int j = 0; j < Col; j++){
                expansion(i, j) = (*this)(i, j);
            }
        }
        //右に単位行列を代入
        for(int i = Col; i < Col * 2; i++){
            expansion(i - Row, i) = 1;
        }

        for(int i = 0; i < Row; i++){
            //ピボットの選択
            int pivot = i;
            for(int j = i + 1; j < Row; j++){
                if(abs(expansion(j, i)) > abs(expansion(pivot, i)))
                    pivot = j;
            }

            //行の入れ替え
            if(i != pivot){
                for(int j = 0; j < Col * 2; j++){
                    samp = expansion(i, j);
                    expansion(i, j) = expansion(pivot, j);
                    expansion(pivot, j) = samp;
                }
            }

            //左を単位行列に変換
            samp = expansion(i, i);
            for(int j = 0; j < Col * 2; j++){
                expansion(i, j) /= samp;
            }
            for(int j = 0; j < Row; j++){
                if(j != i){
                    buf = expansion(j, i);
                    for(int k = 0; k < Col * 2; k++){
                        expansion(j, k) -= buf * expansion(i, k);
                    }
                }
            }
        }
        //結果の抽出
        for(int i = 0; i < Row; i++){
            for(int j = Col; j < Col * 2; j++){
                result(i, j - Col) = expansion(i, j);
            }
        }
        return result;
    }

    size_t row(){
        return Row;
    }

    size_t col(){
        return Col;
    }

    Vector<int, 2> size(){
        Vector<int, 2> S({Row, Col});
        return S;
    }

    void print() const {
        for (int i = 0; i < Row; ++i) {
            matrix[i].print();
        }
    }
};

/*～加算～*/
//Matrix + Matrix
template<typename L, typename R, int Row, int Col>
class Add_Mat{
private:
    const Matrix<L, Row, Col>& _l;
    const Matrix<R, Row, Col>& _r;
public:
    Add_Mat(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) + _r(i, j);
    }
};
template<typename L, typename R, int Row, int Col>
inline Add_Mat<L, R, Row, Col> operator+(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r){
    return Add_Mat<L, R, Row, Col>(l, r);
}
//Matrix + Scalar
template<typename L, typename R, int Row, int Col>
class Add_MatSca{
private:
    const Matrix<L, Row, Col>& _l;
    const R& _r;
public:
    Add_MatSca(const Matrix<L, Row, Col>& l, const R& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) + _r;
    }
};
template<typename L, typename R, int Row, int Col>
inline Add_MatSca<L, R, Row, Col> operator+(const Matrix<L, Row, Col>& l, const R& r){
    return Add_MatSca<L, R, Row, Col>(l, r);
}
template<typename L, typename R, int Row, int Col>
inline Add_MatSca<L, R, Row, Col> operator+(const L& l, const Matrix<R, Row, Col>& r){
    return Add_MatSca<L, R, Row, Col>(r, l);
}

/*～減算～*/
//Matrix - Matrix
template<typename L, typename R, int Row, int Col>
class Sub_Mat{
private:
    const Matrix<L, Row, Col>& _l;
    const Matrix<R, Row, Col>& _r;
public:
    Sub_Mat(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) - _r(i, j);
    }
};
template<typename L, typename R, int Row, int Col>
inline Sub_Mat<L, R, Row, Col> operator-(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r){
    return Sub_Mat<L, R, Row, Col>(l, r);
}
//Matrix - Scalar
template<typename L, typename R, int Row, int Col>
class Sub_MatSca{
private:
    const Matrix<L, Row, Col>& _l;
    const R& _r;
public:
    Sub_MatSca(const Matrix<L, Row, Col>& l, const R& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) - _r;
    }
};
template<typename L, typename R, int Row, int Col>
inline Sub_MatSca<L, R, Row, Col> operator-(const Matrix<L, Row, Col>& l, const R& r){
    return Sub_MatSca<L, R, Row, Col>(l, r);
}
//Scalar - Matrix
template<typename L, typename R, int Row, int Col>
class Sub_ScaMat{
private:
    const L& _l;
    const Matrix<R, Row, Col>& _r;
public:
    Sub_ScaMat(const L& l, const Matrix<R, Row, Col>& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l - _r(i, j);
    }
};
template<typename L, typename R, int Row, int Col>
inline Sub_ScaMat<L, R, Row, Col> operator-(const L& l, const Matrix<R, Row, Col>& r){
    return Sub_ScaMat<L, R, Row, Col>(l, r);
}

/*～乗算～*/
//Matrix * Matrix
template<typename L, typename R, int Row, int Col>
class Mul_Mat{
private:
    const Matrix<L, Row, Col>& _l;
    const Matrix<R, Row, Col>& _r;
public:
    Mul_Mat(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) * _r(i, j);
    }
};
template<typename L, typename R, int Row, int Col>
inline Mul_Mat<L, R, Row, Col> operator*(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r){
    return Mul_Mat<L, R, Row, Col>(l, r);
}
//Matrix * Scalar
template<typename L, typename R, int Row, int Col>
class Mul_MatSca{
private:
    const Matrix<L, Row, Col>& _l;
    const R& _r;
public:
    Mul_MatSca(const Matrix<L, Row, Col>& l, const R& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) * _r;
    }
};
template<typename L, typename R, int Row, int Col>
inline Mul_MatSca<L, R, Row, Col> operator*(const Matrix<L, Row, Col>& l, const R& r){
    return Mul_MatSca<L, R, Row, Col>(l, r);
}
template<typename L, typename R, int Row, int Col>
inline Mul_MatSca<L, R, Row, Col> operator*(const L& l, const Matrix<R, Row, Col>& r){
    return Mul_MatSca<L, R, Row, Col>(r, l);
}

/*～除算～*/
//Matrix / Matrix
template<typename L, typename R, int Row, int Col>
class Div_Mat{
private:
    const Matrix<L, Row, Col>& _l;
    const Matrix<R, Row, Col>& _r;
public:
    Div_Mat(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) / _r(i, j);
    }
};
template<typename L, typename R, int Row, int Col>
inline Div_Mat<L, R, Row, Col> operator/(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r){
    return Div_Mat<L, R, Row, Col>(l, r);
}
//Matrix / Scalar
template<typename L, typename R, int Row, int Col>
class Div_MatSca{
private:
    const Matrix<L, Row, Col>& _l;
    const R& _r;
public:
    Div_MatSca(const Matrix<L, Row, Col>& l, const R& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) / _r;
    }
};
template<typename L, typename R, int Row, int Col>
inline Div_MatSca<L, R, Row, Col> operator/(const Matrix<L, Row, Col>& l, const R& r){
    return Div_MatSca<L, R, Row, Col>(l, r);
}
//Scalar / Matrix
template<typename L, typename R, int Row, int Col>
class Div_ScaMat{
private:
    const L& _l;
    const Matrix<R, Row, Col>& _r;
public:
    Div_ScaMat(const L& l, const Matrix<R, Row, Col>& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l / _r(i, j);
    }
};
template<typename L, typename R, int Row, int Col>
inline Div_ScaMat<L, R, Row, Col> operator/(const L& l, const Matrix<R, Row, Col>& r){
    return Div_ScaMat<L, R, Row, Col>(l, r);
}

/*～剰余～*/
//Matrix % Matrix
template<typename L, typename R, int Row, int Col>
class Mod_Mat{
private:
    const Matrix<L, Row, Col>& _l;
    const Matrix<R, Row, Col>& _r;
public:
    Mod_Mat(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) % _r(i, j);
    }
};
template<typename L, typename R, int Row, int Col>
inline Mod_Mat<L, R, Row, Col> operator%(const Matrix<L, Row, Col>& l, const Matrix<R, Row, Col>& r){
    return Mod_Mat<L, R, Row, Col>(l, r);
}
//Matrix % Scalar
template<typename L, typename R, int Row, int Col>
class Mod_MatSca{
private:
    const Matrix<L, Row, Col>& _l;
    const R& _r;
public:
    Mod_MatSca(const Matrix<L, Row, Col>& l, const R& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l(i, j) % _r;
    }
};
template<typename L, typename R, int Row, int Col>
inline Mod_MatSca<L, R, Row, Col> operator%(const Matrix<L, Row, Col>& l, const R& r){
    return Mod_MatSca<L, R, Row, Col>(l, r);
}
//Scalar % Matrix
template<typename L, typename R, int Row, int Col>
class Mod_ScaMat{
private:
    const L& _l;
    const Matrix<R, Row, Col>& _r;
public:
    Mod_ScaMat(const L& l, const Matrix<R, Row, Col>& r):_l(l), _r(r){}
    double operator()(size_t i, size_t j)const{
        return _l % _r(i, j);
    }
};
template<typename L, typename R, int Row, int Col>
inline Mod_ScaMat<L, R, Row, Col> operator%(const L& l, const Matrix<R, Row, Col>& r){
    return Mod_ScaMat<L, R, Row, Col>(l, r);
}

//行列の積
template<typename T, int Row, int Col, int Com>
inline Matrix<T, Row, Col> dot(const Matrix<T, Row, Com>& A, const Matrix<T, Com, Col>& B){
    Matrix<T, Row, Col> C(0);
    for(int i = 0; i < Row; i++){
        for(int j = 0; j < Col; j++){
            for(int k = 0; k < Com; k++){
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

//累乗
template<typename Type, int Size>
Matrix<Type, Size, Size> pow(const Matrix<Type, Size, Size>& A, int n){
    Matrix<Type, Size, Size> B(0);
    B = A;
    for(int i = 1; i < n; i++){
        B *= A;
    }
    return B;
}